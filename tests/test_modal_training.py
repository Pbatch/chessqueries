"""Modal training: submission returns without waiting, and the job's config, staging and resume guards."""

import importlib.util
import json
import os
import tarfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("modal")
pytest.importorskip("wandb")

from chessqueries.config import get_config  # noqa: E402  (after importorskip)
from chessqueries.train.training_job import ChessQueriesTrainingJob, TrainingConfig  # noqa: E402

SCRIPT = Path(__file__).parents[1] / "scripts" / "train_chessqueries_modal.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("train_chessqueries_modal", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trainer_class(trainer: Mock) -> Mock:
    """A stand-in for the Modal class, whose with_options(...)() gives `trainer`"""
    cls = Mock()
    cls.with_options.return_value = Mock(return_value=trainer)
    return cls


def test_training_submission_saves_call_id_without_waiting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_launcher()
    monkeypatch.chdir(tmp_path)
    call = Mock(object_id="fc-test")
    trainer = Mock()
    trainer.train.spawn.return_value = call
    monkeypatch.setattr(launcher, "ModalChessQueriesTrainer", _trainer_class(trainer))
    monkeypatch.setattr(
        launcher,
        "app",
        Mock(app_id="ap-test", get_dashboard_url=Mock(return_value="https://modal.com/id/ap-test")),
    )
    launcher.main(run_name="vits-test", encoder="vit_small_patch14_dinov2.lvd142m", epochs=3, batch_size=16,
                  unfreeze_epoch=1, gpu="H100")

    launcher.ModalChessQueriesTrainer.with_options.assert_called_once_with(gpu="H100")
    trainer.train.spawn.assert_called_once()
    trainer.train.remote.assert_not_called()
    call.get.assert_not_called()
    config = trainer.train.spawn.call_args.args[0]
    # Every setting the launcher sends must be one the job accepts
    TrainingConfig(**config)
    submission = json.loads((tmp_path / "data/modal-runs/fc-test.json").read_text())
    assert submission["call_id"] == "fc-test"
    assert submission["app_id"] == "ap-test"
    assert submission["run_name"] == "vits-test"
    assert submission["encoder"] == "vit_small_patch14_dinov2.lvd142m"
    assert submission["epochs"] == 3
    assert submission["batch_size"] == 16
    assert submission["gpu"] == "H100"


def test_bad_settings_fail_before_submitting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_launcher()
    monkeypatch.chdir(tmp_path)
    trainer = Mock()
    monkeypatch.setattr(launcher, "ModalChessQueriesTrainer", _trainer_class(trainer))
    # Unfreezing at the default epoch 5 is impossible in a 3-epoch run
    with pytest.raises(ValueError, match="unfreeze"):
        launcher.main(run_name="short", epochs=3)
    trainer.train.spawn.assert_not_called()
    assert not (tmp_path / "data/modal-runs").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"run_name": "../escape"},
        {"epochs": 0},
        {"precision": "8-bit"},
        {"resolution": 644},
        {"learning_rate": 0.0},
        {"unfreeze_epoch": 45},
        {"datasets": "chessred,imagenet"},
        {"augment": "sideways"},
        {"lr_scaling": "cubic"},
    ],
)
def test_config_rejects_bad_settings(change: dict) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**{"dataset": "/datasets/d.tar", "run_name": "run", **change})


@pytest.mark.parametrize(
    ("batch_size", "lr_scaling", "factor"),
    [(6, "sqrt", 1.0), (24, "sqrt", 2.0), (24, "linear", 4.0), (24, "none", 1.0)],
)
def test_learning_rates_scale_from_the_paper_batch(batch_size: int, lr_scaling: str, factor: float) -> None:
    config = TrainingConfig("/datasets/d.tar", "run", batch_size=batch_size, lr_scaling=lr_scaling)
    assert config.lr_factor() == pytest.approx(factor)


def test_stage_datasets_extracts_tar_and_points_loaders_at_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "unused"))
    packed = tmp_path / "pack" / "chessqueries-data"
    (packed / "chesscam").mkdir(parents=True)
    (packed / "chesscam" / "annotations.json").write_text("[]")
    volume = tmp_path / "volume"
    (volume / "datasets").mkdir(parents=True)
    with tarfile.open(volume / "datasets" / "d.tar", "w") as tar:
        tar.add(packed, arcname="chessqueries-data")

    job = ChessQueriesTrainingJob(TrainingConfig("/datasets/d.tar", "run"), volume, Mock())
    staged = job.stage_datasets(tmp_path / "local")
    try:
        assert staged == tmp_path / "local" / "chessqueries-data"
        assert os.environ["DATA_ROOT"] == str(staged)
        assert get_config().chesscam_root == staged / "chesscam"
    finally:
        get_config.cache_clear()


def _write_weights(path: Path, **metadata: str) -> None:
    import torch
    from safetensors.torch import save_file

    # The metadata ChessQ Lite ships with
    defaults = {"format": "chessqueries-safetensors-v1", "encoder_name": "vit_small_patch14_dinov2.lvd142m",
                "resolution": "644", "head_type": "query", "decoder_layers": "4", "nheads": "8", "aux_heads": "false",
                "drop_path_rate": "0.0", "freeze_encoder": "false", "normalization": "imagenet",
                "license": "PolyForm-Noncommercial-1.0.0"}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"x": torch.zeros(1)}, str(path), metadata={**defaults, **metadata})


def test_init_weights_must_match_the_run(tmp_path: Path) -> None:
    _write_weights(tmp_path / "weights" / "lite.safetensors")
    lite = dict(init_weights="/weights/lite.safetensors", encoder="vit_small_patch14_dinov2.lvd142m")

    job = ChessQueriesTrainingJob(TrainingConfig("/datasets/d.tar", "run", **lite), tmp_path, Mock())
    assert job.init_weights_path() == tmp_path / "weights" / "lite.safetensors"

    # 644px weights can be fine-tuned at another resolution
    lower = ChessQueriesTrainingJob(TrainingConfig("/datasets/d.tar", "run", resolution=448, **lite), tmp_path, Mock())
    assert lower.init_weights_path() == tmp_path / "weights" / "lite.safetensors"

    # The default encoder is DINOv3, which the DINOv2 weights can't fill
    wrong = ChessQueriesTrainingJob(TrainingConfig("/datasets/d.tar", "run", init_weights=lite["init_weights"]),
                                    tmp_path, Mock())
    with pytest.raises(ValueError, match="encoder_name"):
        wrong.init_weights_path()

    missing = ChessQueriesTrainingJob(TrainingConfig("/datasets/d.tar", "run", init_weights="/weights/none.safetensors"),
                                      tmp_path, Mock())
    with pytest.raises(FileNotFoundError):
        missing.init_weights_path()


def test_evaluate_only_submits_the_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_launcher()
    monkeypatch.chdir(tmp_path)
    trainer = Mock()
    trainer.evaluate.spawn.return_value = Mock(object_id="fc-eval")
    monkeypatch.setattr(launcher, "ModalChessQueriesTrainer", _trainer_class(trainer))
    monkeypatch.setattr(launcher, "app", Mock(app_id="ap", get_dashboard_url=Mock(return_value="url")))

    with pytest.raises(ValueError, match="init-weights"):
        launcher.main(run_name="baseline", evaluate_only=True)
    launcher.main(run_name="baseline", evaluate_only=True, init_weights="/weights/lite.safetensors")

    trainer.evaluate.spawn.assert_called_once()
    trainer.train.spawn.assert_not_called()
    assert json.loads((tmp_path / "data/modal-runs/fc-eval.json").read_text())["mode"] == "evaluate"


def test_resume_refuses_changed_settings(tmp_path: Path) -> None:
    output = tmp_path / "runs" / "run"
    (output / "checkpoints").mkdir(parents=True)
    torch.save({"global_step": 40, "lr_schedulers": [{"last_epoch": 40}]}, output / "checkpoints" / "last.ckpt")
    original = TrainingConfig("/datasets/d.tar", "run")
    (output / "config.json").write_text(json.dumps(asdict(original)))

    resumed = TrainingConfig("/datasets/d.tar", "run", resolution=640, resume_checkpoint="/runs/run/checkpoints/last.ckpt")
    job = ChessQueriesTrainingJob(resumed, tmp_path, Mock())
    with pytest.raises(ValueError, match="resolution"):
        job.check_resume(tmp_path / "runs/run/checkpoints/last.ckpt", output)

    rebatched = TrainingConfig("/datasets/d.tar", "run", batch_size=32, resume_checkpoint="/runs/run/checkpoints/last.ckpt")
    with pytest.raises(ValueError, match="batch_size"):
        ChessQueriesTrainingJob(rebatched, tmp_path, Mock()).check_resume(tmp_path / "runs/run/checkpoints/last.ckpt",
                                                                         output)

    same = TrainingConfig("/datasets/d.tar", "run", epochs=60, resume_checkpoint="/runs/run/checkpoints/last.ckpt")
    ChessQueriesTrainingJob(same, tmp_path, Mock()).check_resume(tmp_path / "runs/run/checkpoints/last.ckpt", output)


def test_resume_refuses_per_epoch_schedule(tmp_path: Path) -> None:
    output = tmp_path / "runs" / "run"
    (output / "checkpoints").mkdir(parents=True)
    # Written when the schedule stepped per epoch: 4 epochs done, 40 optimizer steps
    torch.save({"global_step": 40, "lr_schedulers": [{"last_epoch": 4}]}, output / "checkpoints" / "last.ckpt")
    config = TrainingConfig("/datasets/d.tar", "run", resume_checkpoint="/runs/run/checkpoints/last.ckpt")
    (output / "config.json").write_text(json.dumps(asdict(config)))
    with pytest.raises(ValueError, match="per-epoch"):
        ChessQueriesTrainingJob(config, tmp_path, Mock()).check_resume(output / "checkpoints" / "last.ckpt", output)


def test_resolution_follows_encoder_patch_size() -> None:
    dinov3 = "vit_small_patch16_dinov3.lvd1689m"
    TrainingConfig("/datasets/d.tar", "run", encoder=dinov3, resolution=640)
    with pytest.raises(ValueError, match="16px"):
        TrainingConfig("/datasets/d.tar", "run", encoder=dinov3, resolution=644)
    with pytest.raises(ValueError, match="14px"):
        TrainingConfig("/datasets/d.tar", "run", encoder="vit_large_patch14_dinov2.lvd142m", resolution=640)

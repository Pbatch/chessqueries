"""Class-based orchestration for ChessQueries training on a Modal volume, with logging and export.

The datasets arrive as one tar (scripts/pack_datasets.py) on the volume. It is extracted to the
container's local disk, since shuffled reads from a network volume are slow, and DATA_ROOT is
pointed at it. Runs write to <volume>/runs/<run_name>/, committing the volume on every checkpoint
so a preempted run can be resumed from its last.ckpt.
"""
from __future__ import annotations

import json
import os
import tarfile
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from chessqueries.config import get_config
from chessqueries.core import Split
from chessqueries.data.base import DatasetName, get_dataset
from chessqueries.data.transforms import Augment, build_transform
from chessqueries.metrics import aggregate
from chessqueries.models.base import predict_all
from chessqueries.models.checkpoint import (
    export_lightning_safetensors,
    load_safetensors_model,
    read_safetensors_metadata,
)
from chessqueries.models.chessqueries_model import HeadType
from chessqueries.train.baseline_recipes import Schedule
from chessqueries.train.lit import JointDataModule, LitChessQueriesModel
from chessqueries.train.training_logging import create_wandb_logger

PRECISIONS = {"32-true", "16-mixed", "bf16-mixed"}
# How the learning rates grow with the batch size, relative to the paper's batch of 6
LR_SCALINGS = {"none", "sqrt", "linear"}
PAPER_BATCH_SIZE = 6
# Settings that must not change when resuming: the checkpoint can't record them all, and a new
# batch size would rescale the learning rates mid-schedule
RESUME_FIXED = ("encoder", "resolution", "datasets", "head_type", "decoder_layers", "batch_size", "lr_scaling")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Defaults are the paper recipe (ViT-L/14 at 644px on ChessReD + ChessCog, plus ChessCam),
    except the batch: the paper's 6 leaves an A100-80GB mostly idle, so it is 16 here, with the
    paper's learning rates (given for batch 6) scaled by the square root of the batch ratio."""

    dataset: str
    run_name: str
    epochs: int = 45
    batch_size: int = 16
    lr_scaling: str = "sqrt"
    resume_checkpoint: str = ""
    precision: str = "bf16-mixed"
    resolution: int = 644
    encoder: str = "vit_large_patch14_dinov2.lvd142m"
    datasets: str = "chessred,chesscog,chesscam"
    learning_rate: float = 1.4e-4
    encoder_learning_rate: float = 1.4e-5
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    unfreeze_epoch: int = 5
    lr_floor: float = 0.001
    augment: str = Augment.GEOMETRIC_MILD.value
    # Augment on the GPU per batch; loader workers on CPU then only decode and resize
    gpu_augment: bool = True
    head_type: str = HeadType.QUERY.value
    decoder_layers: int = 4
    seed: int = 1
    workers: int = 8
    # Start from released weights (a .safetensors on the volume) instead of DINOv2 alone
    init_weights: str = ""
    # Test splits the last weights are scored on after training
    eval_datasets: str = "chesscam,chessred,chesscog"
    wandb_project: str = "chessqueries"
    wandb_entity: str = ""

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("Use positive epochs and batch size")
        if not self.run_name or Path(self.run_name).name != self.run_name:
            raise ValueError("Use a simple run name")
        if self.precision not in PRECISIONS:
            raise ValueError(f"Precision must be one of {sorted(PRECISIONS)}")
        if self.resolution % 14:
            raise ValueError("Resolution must be a multiple of the 14px ViT patch size")
        if not 0 < self.learning_rate < float("inf") or not 0 < self.encoder_learning_rate < float("inf"):
            raise ValueError("Learning rates must be positive and finite")
        if not 0 <= self.unfreeze_epoch < self.epochs:
            raise ValueError("The encoder must unfreeze before the last epoch")
        if self.lr_scaling not in LR_SCALINGS:
            raise ValueError(f"LR scaling must be one of {sorted(LR_SCALINGS)}")
        Augment(self.augment)
        HeadType(self.head_type)
        self.sources()
        self.eval_sources()

    def sources(self) -> tuple[DatasetName, ...]:
        return tuple(DatasetName(name.strip()) for name in self.datasets.split(",") if name.strip())

    def eval_sources(self) -> tuple[DatasetName, ...]:
        return tuple(DatasetName(name.strip()) for name in self.eval_datasets.split(",") if name.strip())

    def lr_factor(self) -> float:
        """Multiplier on the learning rates, which are given for the paper's batch of 6"""
        ratio = self.batch_size / PAPER_BATCH_SIZE
        return {"none": 1.0, "sqrt": ratio ** 0.5, "linear": ratio}[self.lr_scaling]


def check_weights_match(path: Path, config: TrainingConfig) -> None:
    """Released weights only load into a model built the same way"""
    metadata = read_safetensors_metadata(path)
    expected = {
        "encoder_name": config.encoder,
        "resolution": str(config.resolution),
        "head_type": config.head_type,
        "decoder_layers": str(config.decoder_layers),
    }
    mismatched = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatched:
        raise ValueError(f"{path} doesn't match the run's settings (weights, run): {mismatched}")


@torch.no_grad()
def evaluate_on_test(model: torch.nn.Module, config: TrainingConfig) -> dict[str, dict[str, float]]:
    """Exact-board and per-square accuracy of `model` on each eval dataset's test split"""
    model = model.to("cuda").eval()
    transform = build_transform(config.resolution, train=False)
    results = {}
    for name in config.eval_sources():
        dataset = get_dataset(name).torch_dataset(Split.TEST, transform=transform)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = predict_all(model, dataset, device="cuda", batch_size=32, workers=config.workers,
                                      desc=f"test {name.value}")
        metrics = aggregate(predictions.preds, predictions.gts)
        results[name.value] = {key: metrics[key] for key in ("n_boards", "board_accuracy", "per_square_accuracy")}
        print(name.value, results[name.value])
    return results


class VolumeCheckpoint(ModelCheckpoint):
    """Keeps the best (by validation board accuracy) and last checkpoints, committing each save"""

    def __init__(self, output: Path, commit: Callable[[], None]) -> None:
        super().__init__(
            dirpath=output / "checkpoints",
            filename="best",
            monitor="val_board_acc",
            mode="max",
            save_top_k=1,
            save_last=True,
            enable_version_counter=False,
        )
        self.commit = commit

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        self.commit()


@dataclass(slots=True)
class ChessQueriesTrainingJob:
    config: TrainingConfig
    root: Path
    commit: Callable[[], None]

    def stage_datasets(self, directory: Path) -> Path:
        """Extract the dataset tar to local disk and point the dataset loaders at it"""
        tar_path = self.root / self.config.dataset.lstrip("/")
        if not tar_path.is_file():
            raise FileNotFoundError(tar_path)
        with tarfile.open(tar_path) as tar:
            tar.extractall(directory, filter="data")
        roots = [p for p in directory.iterdir() if p.is_dir()]
        if len(roots) != 1:
            raise ValueError(f"Expected one top-level directory in {tar_path}, found {roots}")
        os.environ["DATA_ROOT"] = str(roots[0])
        get_config.cache_clear()  # the loaders read DATA_ROOT through the cached settings
        return roots[0]

    def check_resume(self, checkpoint: Path, output: Path) -> None:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        saved_path = output / "config.json"
        if not saved_path.is_file():
            raise FileNotFoundError(f"{saved_path} is missing, so the run's settings can't be checked")
        saved = json.loads(saved_path.read_text())
        changed = [key for key in RESUME_FIXED if saved.get(key) != getattr(self.config, key)]
        if changed:
            raise ValueError(f"Resume must keep the original {', '.join(changed)}")

    def init_weights_path(self) -> Path | None:
        if not self.config.init_weights:
            return None
        path = self.root / self.config.init_weights.lstrip("/")
        if not path.is_file():
            raise FileNotFoundError(path)
        check_weights_match(path, self.config)
        return path

    def evaluate_only(self) -> dict[str, dict[str, float]]:
        """Score `init_weights` as they are, with the same test code as a trained run (a baseline)"""
        weights = self.init_weights_path()
        if weights is None:
            raise ValueError("Evaluation needs init_weights")
        output = self.root / "evals" / self.config.run_name
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as temporary:
            self.stage_datasets(Path(temporary))
            results = evaluate_on_test(load_safetensors_model(weights).float(), self.config)
        (output / "test_metrics.json").write_text(json.dumps({"weights": self.config.init_weights, **results},
                                                             indent=2) + "\n")
        self.commit()
        return results

    def run(self) -> None:
        config = self.config
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("wandb-secret must contain WANDB_API_KEY")
        output = self.root / "runs" / config.run_name
        checkpoint = self.root / config.resume_checkpoint.lstrip("/") if config.resume_checkpoint else None
        if checkpoint is not None:
            self.check_resume(checkpoint, output)
        # A resumed run takes its weights from the checkpoint instead
        weights = self.init_weights_path() if checkpoint is None else None
        output.mkdir(parents=True, exist_ok=checkpoint is not None)
        (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
        pl.seed_everything(config.seed, workers=True)
        torch.set_float32_matmul_precision("high")

        with tempfile.TemporaryDirectory() as temporary:
            data_root = self.stage_datasets(Path(temporary))
            datamodule = JointDataModule(
                config.sources(), config.resolution, config.batch_size, config.workers, augment=config.augment,
                gpu_augment=config.gpu_augment,
            )
            model = LitChessQueriesModel(
                encoder_name=config.encoder,
                freeze_encoder=False,
                decoder_layers=config.decoder_layers,
                lr=config.learning_rate * config.lr_factor(),
                encoder_lr=config.encoder_learning_rate * config.lr_factor(),
                weight_decay=config.weight_decay,
                lr_schedule=Schedule.COSINE,
                warmup_epochs=config.warmup_epochs,
                unfreeze_epoch=config.unfreeze_epoch,
                lr_floor=config.lr_floor,
                head_type=config.head_type,
            )
            if weights is not None:
                model.model.load_state_dict(load_safetensors_model(weights).float().state_dict())
            callback = VolumeCheckpoint(output, self.commit)
            wandb_logger = create_wandb_logger(
                output, run_name=config.run_name, project=config.wandb_project, entity=config.wandb_entity or None
            )
            self.commit()
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                max_epochs=config.epochs,
                precision=config.precision,
                gradient_clip_val=1.0,
                log_every_n_steps=20,
                callbacks=[callback, LearningRateMonitor(logging_interval="epoch")],
                logger=[CSVLogger(str(output), name="logs"), wandb_logger],
                default_root_dir=output,
            )
            with wandb_logger.experiment:
                wandb_logger.log_hyperparams({
                    **asdict(config),
                    "data_root": str(data_root),
                    "lr_factor": config.lr_factor(),
                    "effective_learning_rate": config.learning_rate * config.lr_factor(),
                    "effective_encoder_learning_rate": config.encoder_learning_rate * config.lr_factor(),
                })
                trainer.fit(model, datamodule, ckpt_path=str(checkpoint) if checkpoint is not None else None)
                # Tensor-only copies of the best and last weights, loadable by chessqueries-predict --checkpoint
                for kind, path in (("best", callback.best_model_path), ("last", callback.last_model_path)):
                    export_lightning_safetensors(Path(path), output / f"{kind}.safetensors",
                                                 resolution=config.resolution)
                self.commit()
                # The last weights, not the best: best is picked on a validation set ChessReD dominates
                results = evaluate_on_test(model.model, config)
                (output / "test_metrics.json").write_text(json.dumps({"weights": "last", **results}, indent=2) + "\n")
                wandb_logger.experiment.summary.update({
                    f"test/{name}/{metric}": value
                    for name, metrics in results.items() for metric, value in metrics.items()
                })
                self.commit()

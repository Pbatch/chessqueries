"""Checkpoints with enum hparams survive torch>=2.6 weights_only loading."""

import sys
from enum import Enum
from types import ModuleType

import pytest
import torch

from chessqueries.data.base import DatasetName
from chessqueries.data.transforms import Augment, Normalization
from chessqueries.models.chessqueries_model import HeadType
from chessqueries.train import lit  # noqa: F401  (import registers the globals)
from chessqueries.train.lit import LEGACY_HEAD_TYPE_PATH, LEGACY_SCHEDULE_PATH, Schedule

HPARAM_ENUMS = {Augment, DatasetName, HeadType, Normalization, Schedule}


def test_hparam_enums_are_weights_only_safe():
    safe = set(torch.serialization.get_safe_globals())
    assert HPARAM_ENUMS <= safe


def test_enum_hparams_round_trip_weights_only(tmp_path):
    path = tmp_path / "hparams.ckpt"
    torch.save({"lr_schedule": Schedule.COSINE, "head_type": HeadType.LINEAR}, path)
    loaded = torch.load(path, weights_only=True)
    assert loaded["lr_schedule"] is Schedule.COSINE
    assert loaded["head_type"] is HeadType.LINEAR


@pytest.mark.parametrize(
    "legacy_path, members, current",
    [
        (LEGACY_HEAD_TYPE_PATH, {"QUERY": "query", "LINEAR": "linear"}, HeadType.LINEAR),
        (
            LEGACY_SCHEDULE_PATH,
            {"NONE": "none", "STEP": "step", "COSINE": "cosine"},
            Schedule.COSINE,
        ),
    ],
)
def test_moved_enum_loads_from_legacy_checkpoint_path(
    tmp_path, monkeypatch, legacy_path, members, current
):
    """Existing checkpoints should not require the retired source module."""
    module_name, _, class_name = legacy_path.rpartition(".")
    legacy_module = ModuleType(module_name)
    legacy_enum = Enum(class_name, members, type=str, module=module_name)  # ty: ignore[mismatched-type-name]
    setattr(legacy_module, class_name, legacy_enum)
    monkeypatch.setitem(sys.modules, module_name, legacy_module)

    path = tmp_path / "legacy-hparams.ckpt"
    torch.save({"value": legacy_enum(current.value)}, path)
    monkeypatch.delitem(sys.modules, module_name)

    loaded = torch.load(path, weights_only=True)
    assert loaded["value"] is current

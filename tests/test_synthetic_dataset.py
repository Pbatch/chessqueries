"""Synthetic render loader (no pixels)."""

import json

from chessqueries.core import Split
from chessqueries.data import DatasetName, Synthetic
from chessqueries.data.base import DATASET_REGISTRY

FEN = "3b4/3B2k1/5pb1/p1pN3p/P1P1P2P/1P3K2/8/8"


def _render(root, split: str, line: str, suffix: str = ".png") -> None:
    folder = root / "render" / split
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{line}.json").write_text(json.dumps({"fen": FEN, "white_turn": False}))
    (folder / f"{line}{suffix}").write_bytes(b"synthetic image")


def test_registered():
    assert DATASET_REGISTRY[DatasetName.SYNTHETIC] is Synthetic


def test_counts_come_from_disk_so_strict_loading_passes(tmp_path):
    _render(tmp_path, "train", "000000")
    _render(tmp_path, "train", "000001", suffix=".jpg")  # a packed copy
    _render(tmp_path, "val", "000002")
    ds = Synthetic(tmp_path)
    assert ds.expected_samples == {Split.TRAIN: 2, Split.VAL: 1, Split.TEST: 0}
    train = ds.load_samples(Split.TRAIN)
    assert [s.image_path.suffix for s in train] == [".png", ".jpg"]
    assert all(s.dataset is DatasetName.SYNTHETIC for s in train)
    assert train[0].board.placement == FEN

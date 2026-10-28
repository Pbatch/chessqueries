"""ChessCam dataset: frames of real games labelled with piece boxes and board x-corners,
converted to FENs by ``make_chessqueries_dataset.py`` in CameraChessLabels."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from chessqueries.config import get_config
from chessqueries.core import Board, Split
from chessqueries.data.base import ChessDataset, DatasetName, BoardSample


class ChessCam(ChessDataset):
    name = DatasetName.CHESSCAM
    splits = (Split.TRAIN, Split.VAL, Split.TEST)

    def __init__(self, dataroot: Path | None = None) -> None:
        self.dataroot = Path(dataroot) if dataroot else get_config().chesscam_root
        self.annotations_path = self.dataroot / "annotations.json"
        if not self.annotations_path.is_file():
            raise FileNotFoundError(
                f"{self.annotations_path} not found; build it with make_chessqueries_dataset.py"
            )
        self._annotations = json.loads(self.annotations_path.read_text())
        # Not frozen like the paper datasets: the conversion decides the counts
        counts = Counter(a["split"] for a in self._annotations)
        self.expected_samples = {split: counts[split.value] for split in self.splits}

    def _load_samples(self, split: Split | None) -> list[BoardSample]:
        assert split is not None  # _validate_split requires one
        return [
            BoardSample(
                image_path=self.dataroot / a["image"],
                board=Board.from_fen(a["fen"]),
                dataset=DatasetName.CHESSCAM,
                sample_id=Path(a["image"]).stem,
                split=split,
                meta={"group": a["group"], "model_agreement": a.get("model_agreement")},
            )
            for a in self._annotations
            if a["split"] == split.value
        ]

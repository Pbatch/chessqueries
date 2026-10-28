"""Our synthetic renders (synthetic/render.py): Blender boards in ChessCog's layout and label format."""

from __future__ import annotations

from pathlib import Path

from chessqueries.config import get_config
from chessqueries.core import Split
from chessqueries.data.base import DatasetName
from chessqueries.data.chesscog import ChessCog


class Synthetic(ChessCog):
    name = DatasetName.SYNTHETIC

    def __init__(self, dataroot: Path | None = None) -> None:
        self.dataroot = Path(dataroot) if dataroot else get_config().synthetic_root
        # Not frozen like ChessCog's: the render run decides the counts
        self.expected_samples = {
            split: sum(1 for _ in (self.dataroot / "render" / split.value).glob("*.json"))
            for split in self.splits
        }

"""Dataset loaders. Every dataset subclasses :class:`ChessDataset` and yields a
common :class:`BoardSample`. Importing this package registers all datasets.
"""
from chessqueries.data.base import (
    BoardImageDataset,
    BoardSample,
    ChessDataset,
    DATASET_REGISTRY,
    DatasetCompleteness,
    DatasetIncompleteError,
    DatasetLoad,
    DatasetName,
    get_dataset,
)
from chessqueries.data.chesscam import ChessCam
from chessqueries.data.chesscog import ChessCog
from chessqueries.data.chessred import ChessReD
from chessqueries.data.cvchess import CVChess, Viewpoint
from chessqueries.data.slcc import SLCC

__all__ = [
    "BoardImageDataset",
    "BoardSample",
    "ChessCam",
    "ChessCog",
    "ChessDataset",
    "ChessReD",
    "CVChess",
    "DatasetCompleteness",
    "DatasetIncompleteError",
    "DatasetLoad",
    "SLCC",
    "Viewpoint",
    "DATASET_REGISTRY",
    "DatasetName",
    "get_dataset",
]

"""FEN-list parsing shared by the Blender renderer and the tests.

Plain Python only: this module is imported inside Blender's bundled
interpreter, which has neither python-chess nor this repo's dependencies.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

PIECE_SYMBOLS = "PNBRQKpnbrqk"
FILES = "abcdefgh"


@dataclass(frozen=True)
class Position:
    placement: str  # FEN piece placement, rank 8 first
    white_to_move: bool | None  # None when the line does not say

    def pieces(self) -> list[tuple[str, int, int]]:
        """``(symbol, file, rank)`` for every piece, both indices 0-based from a1."""
        out = []
        for row_index, row in enumerate(self.placement.split("/")):
            rank = 7 - row_index
            file = 0
            for ch in row:
                if ch.isdigit():
                    file += int(ch)
                else:
                    out.append((ch, file, rank))
                    file += 1
        return out


def square_name(file: int, rank: int) -> str:
    return f"{FILES[file]}{rank + 1}"


def is_valid_placement(placement: str) -> bool:
    rows = placement.split("/")
    if len(rows) != 8:
        return False
    for row in rows:
        width = 0
        for ch in row:
            if ch in "12345678":
                width += int(ch)
            elif ch in PIECE_SYMBOLS:
                width += 1
            else:
                return False
        if width != 8:
            return False
    return True


def parse_position(line: str) -> Position:
    """Parse one line of a FEN list.

    Accepts a full FEN (``<placement> w KQkq - 0 1``), a bare placement, or
    ChessCog's ``fens.txt`` format (``W``/``B`` immediately followed by the
    placement).
    """
    fields = line.split()
    if not fields:
        raise ValueError("empty FEN line")
    token = fields[0]
    if is_valid_placement(token):
        turn = None
        if len(fields) > 1:
            if fields[1] not in ("w", "b"):
                raise ValueError(f"bad side to move {fields[1]!r} in {line!r}")
            turn = fields[1] == "w"
        return Position(token, turn)
    # A leading "B" is also a white bishop on a8, so the prefix is only taken
    # when the line is not already a valid placement on its own
    if token[0] in "WB" and is_valid_placement(token[1:]):
        return Position(token[1:], token[0] == "W")
    raise ValueError(f"not a FEN placement: {line!r}")


def read_positions(path: str) -> list[Position]:
    with open(path) as f:
        return [parse_position(line) for line in f if line.strip()]


def assign_split(placement: str, fractions: dict[str, float]) -> str:
    """Deterministic split from the placement alone, so a position repeated in
    the FEN list never lands in two splits."""
    digest = hashlib.sha256(placement.encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    cumulative = 0.0
    for name, fraction in fractions.items():
        cumulative += fraction
        if u < cumulative:
            return name
    return name

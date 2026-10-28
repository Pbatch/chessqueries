"""Sample positions from PGN games into a FEN list for ``synthetic/render.py``.

Positions come from real games, as in ChessCog, so they are legal and look
like boards people photograph. Each ply is kept independently with
probability ``--rate``; duplicates (same placement and side to move) and the
starting placement, which a game can return to, are dropped.

    uv run --group dev python synthetic/make_fens.py games.pgn --out data/synthetic/fens.txt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import chess
import chess.pgn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pgns", nargs="+", type=Path, help="PGN files to sample from")
    parser.add_argument("--out", type=Path, required=True, help="FEN list to write")
    parser.add_argument("--rate", type=float, default=0.1, help="probability of keeping each ply")
    parser.add_argument("--max", type=int, default=None, help="stop after this many positions")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seen: set[str] = set()
    fens: list[str] = []
    for pgn_path in args.pgns:
        with open(pgn_path, errors="replace") as pgn:
            while (game := chess.pgn.read_game(pgn)) is not None:
                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)
                    if rng.random() >= args.rate:
                        continue
                    key = f"{board.board_fen()} {'w' if board.turn else 'b'}"
                    if key in seen or board.board_fen() == chess.STARTING_BOARD_FEN:
                        continue
                    seen.add(key)
                    fens.append(board.fen())
                if args.max is not None and len(fens) >= args.max:
                    break
        if args.max is not None and len(fens) >= args.max:
            break
    fens = fens[: args.max]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(f + "\n" for f in fens))
    print(f"wrote {len(fens)} positions to {args.out}")


if __name__ == "__main__":
    main()

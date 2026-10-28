"""Download datasets.

Usage:
    uv run python scripts/download_data.py chessred
    uv run python scripts/download_data.py chesscog cvchess
    uv run python scripts/download_data.py all
"""

import argparse

from chessqueries.data import download as dl
from chessqueries.data.base import DatasetName


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "datasets",
        nargs="+",
        choices=[*(d.value for d in dl.DATASETS), "all"],
        help="Which dataset(s) to download.",
    )
    args = p.parse_args(argv)

    names = list(dl.DATASETS) if "all" in args.datasets else [DatasetName(d) for d in args.datasets]
    for name in names:
        dl.DATASETS[name]()

    print("\nDone.")


if __name__ == "__main__":
    main()

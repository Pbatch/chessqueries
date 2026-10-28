"""Evaluate a trained ChessQueries checkpoint on any dataset, zero-shot.

Fills a cell of the cross-dataset reproduction matrix. CVChess has no official
split (pass no --split); ChessCog/ChessReD require one.

With ``--subset-by`` the same inference pass also reports a low/high breakdown of
the headline metrics along a per-sample feature (e.g. crop resolution), so you
can see where the model is weak without a second pass or re-scoring saved preds.

Usage:
    uv run python scripts/eval_cross_dataset.py --checkpoint checkpoints/v0-finetune/last.ckpt \
        --dataset chesscog --split test

    # break test metrics down by crop pixel area (auto threshold = median):
    uv run python scripts/eval_cross_dataset.py --checkpoint .../last.ckpt \
        --dataset slcc --split test --subset-by resolution
"""

import argparse
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from chessqueries.config import get_config
from chessqueries.data import DatasetIncompleteError, DatasetName, get_dataset
from chessqueries.data.transforms import build_transform
from chessqueries.metrics import aggregate, aggregate_subsets
from chessqueries.metrics.report import (
    eval_tag,
    evaluation_inventory,
    print_evaluation_inventory,
    print_metrics,
    resolve_split,
    write_eval_outputs,
)
from chessqueries.models.base import predict_all
from chessqueries.models.checkpoint import load_safetensors_model, read_safetensors_metadata
from chessqueries.models.chessqueries_model import ChessQueriesModel
from chessqueries.train.lit import LitChessQueriesModel


def _resolution(sample) -> float:
    """Crop pixel area (width*height) — the per-square pixel budget the model sees."""
    w, h = Image.open(sample.image_path).size  # lazy: reads header only, not pixels
    return float(w * h)


# Pluggable per-sample numeric features for --subset-by (closed set; extend here).
SUBSET_FEATURES = {"resolution": _resolution}


def _load_model(checkpoint: Path, device: str) -> ChessQueriesModel:
    """A .safetensors export (in its stored precision; FP32 on the CPU) or a Lightning .ckpt."""
    if checkpoint.suffix == ".safetensors":
        model = load_safetensors_model(checkpoint, device=device)
        return model.float() if torch.device(device).type == "cpu" else model
    lit = LitChessQueriesModel.load_from_checkpoint(str(checkpoint), map_location=device)
    return lit.model.eval().to(device)


def _quantile(values: list[float], f: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(f * len(s)))]


def _band_names(n_bands: int) -> list[str]:
    """Human labels for n_bands ordered low->high; named for the common 2/3 cases."""
    if n_bands == 2:
        return ["low", "high"]
    if n_bands == 3:
        return ["low", "medium", "high"]
    return [f"band{i}" for i in range(n_bands)]


@dataclass(frozen=True)
class Banding:
    """Values bucketed into named bands: the band each value fell in, plus the band
    names in low->high order (the order to report them in, including empty bands)."""

    per_value: list[str]
    band_order: list[str]


def _bucketize(values: list[float], thresholds: list[float]) -> Banding:
    """Map each value to a band label given sorted ascending thresholds (k cuts -> k+1 bands)."""
    cuts = sorted(thresholds)
    names = _band_names(len(cuts) + 1)
    return Banding(per_value=[names[bisect_right(cuts, v)] for v in values], band_order=names)


def _quantiles(values: list[float]) -> str:
    return (
        f"min={min(values):.0f}  p10={_quantile(values, 0.1):.0f}  median={_quantile(values, 0.5):.0f}  "
        f"p90={_quantile(values, 0.9):.0f}  max={max(values):.0f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--dataset", required=True, type=DatasetName, choices=list(DatasetName))
    p.add_argument("--split", default=None, choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Input size; defaults to the one a .safetensors checkpoint was trained at.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--name", required=True, help="Run tag; names outputs/chessqueries_<name>.")
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Diagnostic only: evaluate available images from an incomplete dataset and "
        "mark the output non-paper-comparable.",
    )
    p.add_argument(
        "--subset-by",
        choices=list(SUBSET_FEATURES),
        default=None,
        help="Also break the metrics down into bands along this per-sample feature.",
    )
    p.add_argument(
        "--subset-thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Ascending cut(s) on the --subset-by feature; k cuts -> k+1 bands "
        "(1->low/high, 2->low/medium/high). Default = median (one cut).",
    )
    args = p.parse_args()
    if not args.checkpoint.is_file():
        p.error(f"checkpoint not found: {args.checkpoint}")

    if args.resolution is None:
        if args.checkpoint.suffix != ".safetensors":
            p.error(
                "--resolution is required for a Lightning .ckpt (only .safetensors record theirs)"
            )
        args.resolution = int(read_safetensors_metadata(args.checkpoint)["resolution"])

    cfg = get_config()
    ds_obj = get_dataset(args.dataset)
    split = resolve_split(ds_obj, args.split)

    try:
        ds = ds_obj.torch_dataset(
            split,
            transform=build_transform(args.resolution, train=False),
            allow_partial=args.allow_partial,
        )
    except DatasetIncompleteError as exc:
        p.error(str(exc))

    model = _load_model(args.checkpoint, args.device)
    predicted = predict_all(
        model,
        ds,
        device=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
        desc=f"{args.dataset.value}[{split}]",
    )

    metrics = aggregate(predicted.preds, predicted.gts)
    tag = eval_tag(args.dataset.value, split, sep="/")
    print_metrics(metrics, f"ChessQueries ({args.name}) on {tag} | n={len(predicted)}")
    assert ds.completeness is not None  # torch_dataset always records it
    inventory = evaluation_inventory(ds.completeness, evaluated_samples=len(predicted))
    print_evaluation_inventory(inventory)

    if args.subset_by:
        feat = SUBSET_FEATURES[args.subset_by]
        values = [feat(s) for s in ds.samples]
        thresholds = (
            sorted(args.subset_thresholds) if args.subset_thresholds else [_quantile(values, 0.5)]
        )
        banding = _bucketize(values, thresholds)
        breakdown = aggregate_subsets(predicted.preds, predicted.gts, banding.per_value)
        metrics["subset_by"] = {
            "feature": args.subset_by,
            "thresholds": thresholds,
            "subsets": breakdown["subsets"],
        }
        cuts = ", ".join(f"{t:.0f}" for t in thresholds)
        print(f"\n--- by {args.subset_by} (cuts={cuts}) | {_quantiles(values)} ---")
        for band in banding.band_order:
            m = breakdown["subsets"].get(band)
            line = (
                f"board={m['board_accuracy']:.4f}  wrong={m['mean_wrong_squares']:.3f}  (n={m['n_boards']})"
                if m
                else "(empty)"
            )
            print(f"  {band:6s} {line}")

    metrics["evaluation"] = inventory
    suffix = eval_tag(args.dataset.value, split)
    out_dir = write_eval_outputs(
        cfg.OUTPUTS_ROOT / f"chessqueries_{args.name}", suffix, metrics, ds.samples, predicted.preds
    )
    print(f"\nWrote metrics + predictions to {out_dir} ({suffix})")


if __name__ == "__main__":
    main()

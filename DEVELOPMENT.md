# Development

This repository holds the ChessQueries training, evaluation and export code.
Run all commands from the repository root.

## Environment

ChessQueries uses Python 3.12 and uv with an in-project `.venv`.

```bash
# Model, data loaders, evaluation, plus Modal training and tests
uv sync --group modal --group dev
```

| Group | Purpose |
|---|---|
| `modal` | Submitting training runs to Modal and logging them to W&B |
| `dev` | Tests, linting, and pre-commit checks |

`uv sync` makes `.venv` match exactly the groups you name, so pass every group
you want on each sync (or `--all-groups`); groups left out are uninstalled.

## Data

```bash
uv run python scripts/download_data.py all
```

| Dataset | Splits used here | Acquisition |
|---|---|---|
| ChessReD | train/validation/test; 2,129 test images | automatic, about 2.5 GB |
| ChessCog | train/validation/test; 342 test images | automatic |
| ChessCam | train/validation/test; 2,275 test images | built outside this repo, under `data/chesscam/` |
| CVChess | evaluation only; 352 images from one game | labels automatic; images manual |
| SLCC | 1,475/326/373 train/validation/test frames | reconstructed with the upstream ChessQueries tooling, under `data/slcc/dataset/` |

The CVChess downloader seeds `annotations.json` and prints instructions for
placing its manually downloaded images under `data/cvchess/images/`.

## Training

Training runs on Modal. Pack the training datasets into one tar and upload it
once:

```bash
.venv/bin/python scripts/pack_datasets.py
.venv/bin/modal volume put chessqueries-training dist/chessqueries-data.tar /datasets/chessqueries-data.tar
```

then submit a run (DINOv3 ViT-S/16 at 448 pixels by default):

```bash
.venv/bin/modal run --detach scripts/train_chessqueries_modal.py --run-name my-run
```

Run the launcher with `--help` for the encoder, resolution, schedule,
fine-tuning (`--init-weights`) and resume options. Each run writes
`best.safetensors`, `last.safetensors` and `test_metrics.json` under
`/runs/<run-name>/` on the volume.

## Evaluation

```bash
# One domain
uv run python scripts/eval_cross_dataset.py \
  --checkpoint last.safetensors --dataset chesscam --split test --name my-run

# Every domain
bash scripts/eval_all_domains.sh last.safetensors my-run
```

A `.safetensors` checkpoint records its training resolution, which evaluation
uses by default; a Lightning `.ckpt` needs `--resolution`. Evaluation writes
`outputs/chessqueries_<name>/metrics_<domain>.json`. It checks expected dataset
sizes, available images and non-empty splits before running. `--allow-partial`
is only for debugging incomplete local data; its results are marked partial.

## Export

Both exporters take a `.safetensors` checkpoint and run outside the project
environment, since their converters pin their own packages; see each script's
docstring for the full command.

| Format | Script | Used by |
|---|---|---|
| LiteRT (`.tflite`, uint8 RGBA input) | `scripts/export_litert.py` | the ChessCam web app (LiteRT.js on WebGPU) |
| ONNX | `scripts/export_onnx.py` | onnxruntime-web |

## Checks

```bash
uv sync --group dev
uv run ruff check .
uv run pytest
uv run pre-commit run --all-files
```

CI runs Ruff and the full test suite on Python 3.12.

## Repository layout

```text
chessqueries/
  core/               chess types, FEN ordering, and square vocabulary
  data/               dataset loaders, transforms, GPU augmentation, and downloaders
  models/             the ChessQueries model and its safetensors checkpoint format
  train/              Lightning module, multi-dataset input pipeline, and the Modal job
  metrics/            board accuracy, tolerance curve, and edit distance
scripts/               download, packing, training, evaluation, and export entrypoints
synthetic/             Blender renderer for ChessCog-style synthetic boards (see its README)
tests/                 unit and integration tests
```

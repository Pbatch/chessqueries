**♟️ Update: now used on the 2nd-biggest chess website, the fully open-source and non-profit [lichess.org](lichess.org) (see announcement [here](https://lichess.org/feed#dfs7OI))**

# ChessQueries

Chess board recognition: provide a photo of a real-life chessboard and receive
its chess position.

🌐 [Demo](https://chessq.org) ·
📄 [Paper](https://arxiv.org/abs/2608.30762) ·
📊 [Dataset](https://huggingface.co/datasets/joelseytre/slcc) ·
🤖 [Model weights](https://huggingface.co/joelseytre/chessqueries)

[Try it online](https://chessq.org). The website uses a ChessQueries model further
trained on proprietary data.

![ChessQueries predictions across ChessReD, ChessCog, SLCC, and CVChess](assets/readme/hero.jpg)

## Model

ChessQueries consists of a 304M-parameter ViT-L/14 encoder that maps a 644 × 644
input image to patch tokens. Sixty-four learned queries—one for each board
square—cross-attend to those tokens through a DETR-style decoder. A single
shared linear head maps each decoded query to its per-square class.

Best exact-board accuracy reported in the paper for a model trained on the
ChessReD, ChessCog, and SLCC training splits:

| ChessReD | ChessCog | SLCC | CVChess (zero-shot) |
|---:|---:|---:|---:|
| **99.5%** | **98.5%** | **87.1%** | **87.6%** |

## This repository

This repository holds the code to train ChessQueries models (on Modal), evaluate
them across ChessReD, ChessCog, ChessCam, SLCC and CVChess, and export them to
LiteRT or ONNX for the browser. It requires Python 3.12 and
[uv](https://docs.astral.sh/uv/). See [DEVELOPMENT.md](DEVELOPMENT.md) for the
commands.

For released checkpoints, see the Hugging Face pages for
[paper models](https://huggingface.co/joelseytre/chessqueries) and
[ChessQ Lite](https://huggingface.co/joelseytre/chessq-lite).

## Citation

```bibtex
@misc{seytre2026chessqueries,
  title         = {ChessQueries: Toward Better Chess Board Recognition},
  author        = {Seytre, Jo{\"e}l},
  year          = {2026},
  eprint        = {2608.30762},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2608.30762},
  url           = {https://arxiv.org/abs/2608.30762}
}
```

## Licenses

- **Code** — [PolyForm Noncommercial 1.0.0](LICENSE), including the required
  notice: Copyright (c) 2026 the chessqueries authors.
- **Model weights** — [PolyForm Noncommercial 1.0.0](https://huggingface.co/joelseytre/chessqueries#license).
- **SLCC annotations** — [CC BY-NC 4.0](DATA_LICENSE). The source broadcasts
  and reconstructed frames are not distributed and remain the property of
  their respective rights holders.

Commercial use is not licensed.

## Contact

Questions and enquiries: contact "at" chessq.org

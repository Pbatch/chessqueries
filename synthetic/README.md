# Synthetic data

Rendered chessboard photos with labels, made with Blender Cycles. The method
follows ChessCog's synthetic dataset (Wölflein & Arandjelović, 2021). This
version samples a wider range of viewpoints and appearances.

| File | Runs in | Purpose |
|---|---|---|
| `make_fens.py` | project venv (`--group dev`, for python-chess) | samples positions from PGN games into a FEN list |
| `render.py` | Blender, or the `bpy` wheel | renders one image and one JSON label per FEN line |
| `prepare_set.py` | Blender, or the `bpy` wheel | converts a downloaded chess-set model into a `scene.blend` that `render.py` can pick |
| `positions.py` | either | FEN-list parsing and split assignment, shared by the scripts |

## Usage

```bash
# 1. Positions from real games: about 10% of plies, deduplicated, never the
#    starting position
uv run --group dev python synthetic/make_fens.py games.pgn --out data/synthetic/fens.txt

# 2. Render, each FEN on a random set from data/chess_sets/*/scene.blend
blender -b -P synthetic/render.py -- --fens data/synthetic/fens.txt --out data/synthetic
```

Each FEN is rendered on a chess set picked at random from `--sets`, which
defaults to `data/chess_sets`. `--sets` takes `scene.blend` files and
directories holding `*/scene.blend`. The pick depends only on `--seed` and
the FEN's line number, so reruns and shards pick the same set. FENs are
grouped by set, so each `.blend` is opened once per run. The label's
`render.scene` field records which set was used. `--procedural` renders
every FEN on the built-in procedural scene instead.

Written for Blender 5 and tested with Blender 5.2.3. Without a Blender
install, the matching `bpy` wheel from PyPI works the same way. It needs
Python 3.13, so it goes in its own venv:

```bash
uv venv --python 3.13 .venv-bpy && uv pip install --python .venv-bpy/bin/python "bpy==5.2.*"
.venv-bpy/bin/python synthetic/render.py --fens data/synthetic/fens.txt --out data/synthetic
```

Rendering runs on the GPU (OptiX, CUDA, HIP, oneAPI or Metal), and the script
exits if Cycles finds no GPU. To render on the CPU, pass `--device CPU`. At 64
samples and 1200×800, a laptop RTX 3060 renders an image in about 2 s.

Rerunning the script skips images that already have a label, so an
interrupted run picks up where it stopped. To split the work across
processes, pass `--start`/`--end` ranges of FEN lines.

## Output

The output uses ChessCog's layout, `<out>/render/<split>/<line>.{jpg,json}`,
so `ChessCog(dataroot=Path("data/synthetic"))` reads it. Strict loading checks
sample counts against ChessCog's frozen inventory, so pass
`allow_partial=True` to `load_with_report`. To train on the renders as their
own domain, add a dataset class for them.

Splits come from a hash of the piece placement (default 90/3/7). A position
that appears twice in the FEN list therefore always lands in the same split.

Each label holds:

- `fen` (piece placement) and `white_turn`
- `corners`: pixel coordinates of the playing area's corners, in the order
  a1, a8, h8, h1, as in ChessCog
- `pieces`: piece symbol, square and pixel box `[x, y, w, h]` for each piece.
  Boxes come from projecting the mesh vertices, so they include parts that
  other pieces hide.
- `camera`, `lighting`, `exposure` and `looks`: every sampled parameter

## What gets randomised

- **Camera:** elevation 35–75°, from the side of the player to move (azimuth
  σ = 15°, or any direction with probability 0.15), focal length 24–60 mm
  full-frame equivalent, ±3° roll, depth of field in 30% of images.
  Viewpoints are re-sampled until the board corners and every piece lie at
  least 16 px inside the frame.
- **Lighting:** one of camera flash, two spotlights, overhead softbox, or sun.
  Each comes with ambient light, a warm-to-cool colour tint and exposure
  jitter. Light power is set from each light's distance so that every mode
  lands near the same brightness.
- **Scene:** board colour scheme (tournament green, wood, walnut, blue, black
  and white, grey), border width and thickness, wood-like table, piece
  colours and roughness.
- **Pieces:** offset from the square centre (σ = 0.08 squares) and random
  rotation, with overall piece size scaled ±10%. Knights face the opponent
  ±25° in 75% of placements.

ChessCog's spotlights aim at a point drawn in metres where squares were
meant, so they often light empty table. Here the aim point is sampled in
square widths, as their thesis describes.

## Using your own models

To add a chess set, put the downloaded model in its own folder under
`data/chess_sets/` and convert it with `prepare_set.py`. `render.py` picks up
the resulting `scene.blend` automatically:

```bash
python synthetic/prepare_set.py data/chess_sets/<set>/model.glb \
    --out data/chess_sets/<set>/scene.blend --preview
```

`prepare_set.py` needs a model with all 32 pieces standing on the board,
usually in the starting position. It does not rely on how the model is
structured:

- **Pieces:** every mesh is split into its loose parts, and parts are grouped
  into pieces by position.
- **Board geometry:** the piece grid gives the square size, centre and board
  height. The board outline gives its rotation.
- **Colour:** the lighter half of the pieces is white. Brightness is read from
  the actual base colour, including textures and multiply factors. Material
  names are only used when the two halves are too close to tell apart,
  because some models label light-wood pieces "Black".
- **Type:** if all 32 pieces stand in the starting position, their squares
  decide (king and queen by height, in case a model swaps them). Otherwise
  names are used when every piece has one (English, German or Italian), and
  shape decides as a last resort: pawns are the 8 shortest, then king and
  queen by height, knights by their off-axis heads, and rooks are shorter
  than bishops.
- **Board squares:** the board's own colours are fitted with a checker
  pattern. When the fit is clear, the board's squares set the final scale
  and centre, and the board is turned 90° if h1 would come out dark. Some
  models have the board a quarter turn off from the pieces.

The output holds the board, twelve hidden piece templates scaled to
`--square-size` (default 0.055 m) with knights facing +y, and a
`square_size` property that `render.py` reads. Next to it go a JSON report
(method used, counts, warnings) and, with `--preview`, a render of the
starting position. Check the preview: wrong colours or piece types show up
at once. If white ends up on the wrong side, rerun with `--rotate 180`. If
the colours come out swapped, rerun with `--swap-colours`. `--verbose` lists
every detected piece.

A hand-made `.blend` works too if it follows the same layout:

- The board is centred on the origin with its top at z = 0, and white sits on
  the −y side.
- Piece templates are named `White Pawn` … `Black King`, with each origin at
  the base of the piece.

`--square-size` overrides each file's `square_size` property, and 0.05 m is
used when a file has none.

Materials, board and table stay as authored. Only the camera, the added
lights and the piece placement are randomised.

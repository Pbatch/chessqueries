"""Turn a downloaded chess-set model into a ``render.py --scene`` file.

Sketchfab-style models differ in units, axes, naming and how pieces are split
into objects, so nothing about their structure is trusted. Every mesh is cut
into its loose parts, parts are grouped into pieces by position, and the
piece grid gives the square size, board centre, board height and rotation.
Piece colour and type come from object and material names when every piece
has one, otherwise from brightness (lighter half is white) and shape (pawns
are the 8 shortest, then king and queen by height, knights by their
asymmetric heads, rooks shorter than bishops).

The output .blend holds the board (every part that is not a piece), twelve
hidden templates named "White Pawn" ... "Black King" with their origins at
the centre of the base and knights facing +y, and a ``square_size`` scene
property that ``render.py`` reads. A JSON report and, with ``--preview``, a
render of the starting position are written next to it so the mapping can be
checked by eye.

    python synthetic/prepare_set.py data/chess_sets/<set>/model.glb \\
        --out data/chess_sets/<set>/scene.blend --preview

Runs under Blender (``blender -b -P synthetic/prepare_set.py -- ...``) or the
``bpy`` wheel, like ``render.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import bpy  # first: the PyPI wheel only exposes bmesh and mathutils once bpy is imported
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from mathutils.interpolate import poly_3d_calc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from positions import parse_position  # noqa: E402
from render import PIECE_NAMES, look_at, setup_device  # noqa: E402

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

# Checked in this order, so "koenigin" is a queen and "knight" a knight before
# either can match king. ".{1,3}" also covers mangled umlauts ("k��nig")
TYPE_PATTERNS = [
    ("q", r"queen|k.{1,3}nigin|dame|regina"),
    ("n", r"knight|springer|horse|pferd|cavallo"),
    ("k", r"king|k.{1,3}nig"),
    ("r", r"rook|turm|torre|castle"),
    ("b", r"bishop|l(ae|ä)ufer|alfiere"),
    ("p", r"pawn|bauer|pedone"),
]
COLOUR_PATTERNS = {
    "white": r"white|wei(ss|ß)|bianc|blanc|blank|light",
    "black": r"black|schwarz|nero|scuro|dark|dunkel",
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a chess-set model for render.py --scene.")
    parser.add_argument("model", type=Path, help=".glb/.gltf, .fbx, .obj or .blend")
    parser.add_argument("--out", type=Path, required=True, help=".blend to write")
    parser.add_argument(
        "--square-size", type=float, default=0.055, help="square width in metres after rescaling"
    )
    parser.add_argument(
        "--rotate",
        type=float,
        default=0.0,
        help="extra rotation about z in degrees, if white ends up on the wrong side",
    )
    parser.add_argument(
        "--swap-colours",
        action="store_true",
        help="swap white and black, if the brightness guess is wrong",
    )
    parser.add_argument(
        "--board-turn",
        type=float,
        default=None,
        help="rotate only the board by this many degrees, overriding the a1-is-dark check",
    )
    parser.add_argument("--preview", action="store_true", help="also render the starting position")
    parser.add_argument(
        "--device", default="auto", help="Cycles device for --preview (see render.py)"
    )
    parser.add_argument("--verbose", action="store_true", help="list every detected piece")
    return parser.parse_args(argv)


# -- loading and splitting -----------------------------------------------------------


def load_model(path: Path) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    suffix = path.suffix.lower()
    if suffix in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path), merge_vertices=True)
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif suffix == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(path))
    else:
        raise SystemExit(f"unsupported model format {suffix}")


def bake_world_transforms() -> list[bpy.types.Object]:
    """Move every mesh into world coordinates with identity transforms and no
    parents, and drop everything that is not a mesh with faces."""
    objects = list(bpy.context.scene.objects)
    world = {o: o.matrix_world.copy() for o in objects}
    meshes = []
    for o in objects:
        if o.type != "MESH" or len(o.data.polygons) == 0:
            continue
        if o.data.users > 1:
            o.data = o.data.copy()
        o.data.transform(world[o])
        meshes.append(o)
    for o in objects:
        if o not in meshes:
            bpy.data.objects.remove(o, do_unlink=True)
    for o in meshes:
        o.parent = None
        o.matrix_world = Matrix.Identity(4)
        o.hide_render = o.hide_viewport = False
        o.hide_set(False)
    return meshes


def separate_loose(meshes: list[bpy.types.Object]) -> list[bpy.types.Object]:
    view_layer = bpy.context.view_layer
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    view_layer.objects.active = meshes[0]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def join(objects: list[bpy.types.Object], name: str) -> bpy.types.Object:
    if len(objects) > 1:
        with bpy.context.temp_override(
            active_object=objects[0], selected_objects=objects, selected_editable_objects=objects
        ):
            bpy.ops.object.join()
    objects[0].name = name
    return objects[0]


def vertices(obj: bpy.types.Object) -> np.ndarray:
    co = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


# -- piece detection -----------------------------------------------------------------


class Part:
    def __init__(self, obj: bpy.types.Object) -> None:
        self.obj = obj
        v = vertices(obj)
        self.lo, self.hi = v.min(axis=0), v.max(axis=0)
        self.centre = (self.lo + self.hi) / 2
        self.footprint = max(self.hi[0] - self.lo[0], self.hi[1] - self.lo[1])
        self.height = self.hi[2] - self.lo[2]


class Piece:
    def __init__(self, parts: list[Part]) -> None:
        self.parts = parts
        v = np.concatenate([vertices(p.obj) for p in parts])
        self.lo, self.hi = v.min(axis=0), v.max(axis=0)
        self.height = self.hi[2] - self.lo[2]
        # Centre of the base, not of the bounding box, which a knight's head skews
        base = v[v[:, 2] < self.lo[2] + 0.1 * self.height]
        self.base = (base.min(axis=0)[:2] + base.max(axis=0)[:2]) / 2
        top = v[v[:, 2] > self.lo[2] + 0.6 * self.height][:, :2]
        eig = np.linalg.eigvalsh(np.cov((top - top.mean(axis=0)).T)) if len(top) > 3 else (1, 1)
        self.elongation = float(math.sqrt(max(eig) / max(min(eig), 1e-12)))
        # How far the top sits off the base axis, relative to the base width:
        # large for a knight's head, near zero for turned pieces
        width = max(base.max(axis=0)[:2] - base.min(axis=0)[:2])
        self.head_offset = (
            float(np.linalg.norm(top.mean(axis=0) - self.base) / width) if len(top) else 0.0
        )
        self.square: tuple[int, int] | None = None
        self.text = " ".join(
            [p.obj.name for p in parts]
            + [s.material.name for p in parts for s in p.obj.material_slots if s.material]
        ).lower()
        self.colour: str | None = None
        self.type: str | None = None


def cluster(parts: list[Part], radius: float) -> list[list[Part]]:
    """Group parts whose footprint centres are chained within ``radius``."""
    centres = np.array([p.centre[:2] for p in parts])
    parent = list(range(len(parts)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    dist = np.linalg.norm(centres[:, None] - centres[None], axis=2)
    for i, j in zip(*np.nonzero(np.triu(dist < radius, 1))):
        parent[root(i)] = root(j)
    groups: dict[int, list[Part]] = {}
    for i, p in enumerate(parts):
        groups.setdefault(root(i), []).append(p)
    return list(groups.values())


def find_pieces(parts: list[Part]) -> tuple[list[Piece], list[Part], float, float]:
    """Group parts into pieces. Returns the pieces, the remaining parts, a
    rough square size and the board-top height."""
    rough = max(p.footprint for p in parts) / 9  # the largest part is the board or its base
    small = [p for p in parts if p.footprint < 1.3 * rough and p.height > 0.02 * rough]
    if not small:
        raise SystemExit("found no piece-sized parts")
    # Each tall column of parts is a piece resting on the board, so the board
    # top is the median column bottom. Counting columns rather than parts
    # keeps pieces made of many parts from outvoting the rest
    columns = cluster(small, 0.35 * rough)
    bottoms = [
        min(p.lo[2] for p in c)
        for c in columns
        if max(p.hi[2] for p in c) - min(p.lo[2] for p in c) > 0.4 * rough
    ]
    if not bottoms:
        raise SystemExit("found no piece-sized parts")
    board_top = float(np.median(bottoms))

    candidates = [
        p
        for p in small
        if p.lo[2] > board_top - 0.05 * rough and p.hi[2] > board_top + 0.08 * rough
    ]
    pieces = [Piece(g) for g in cluster(candidates, 0.35 * rough)]
    pieces = [p for p in pieces if p.height > 0.4 * rough and p.lo[2] < board_top + 0.15 * rough]

    # Flat bits such as felt pads were too thin to count above; give each
    # leftover small part standing on the board to the piece over it
    used = {id(part) for piece in pieces for part in piece.parts}
    bases = np.array([p.base for p in pieces])
    gaps = np.linalg.norm(bases[:, None] - bases[None], axis=2)
    np.fill_diagonal(gaps, np.inf)
    spacing = float(np.median(gaps.min(axis=1)))
    extra: dict[int, list[Part]] = {}
    for part in parts:
        if (
            id(part) in used
            or part.footprint >= 1.3 * rough
            or part.lo[2] < board_top - 0.05 * rough
        ):
            continue
        # Clearly narrower than a square, so board squares are never taken
        dist = np.linalg.norm(bases - part.centre[:2], axis=1)
        if dist.min() < 0.35 * rough and part.footprint < 0.9 * spacing:
            extra.setdefault(int(dist.argmin()), []).append(part)
            used.add(id(part))
    pieces = [Piece(p.parts + extra.get(i, [])) for i, p in enumerate(pieces)]
    rest = [p for p in parts if id(p) not in used]
    return pieces, rest, rough, board_top


# -- colour and type -----------------------------------------------------------------


def _principled(mat: bpy.types.Material):
    if mat.node_tree is None:
        return None
    return next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)


def _socket_rgb(socket, uv: np.ndarray | None, cache: dict, depth: int = 0) -> np.ndarray:
    """Mean colour reaching ``socket``, following image textures (sampled at
    ``uv``), multiply/mix nodes and flat colours. Anything else passes its
    first colour input through."""
    if not socket.is_linked or depth > 8:
        value = socket.default_value
        return np.array(value[:3] if hasattr(value, "__len__") else (value,) * 3, float)
    node = socket.links[0].from_node
    if node.type == "TEX_IMAGE" and node.image is not None:
        px = _image_array(node.image, cache)
        if uv is None:
            return px[..., :3].reshape(-1, 3).mean(axis=0)
        xy = (np.mod(uv, 1) * 127).astype(int)
        return px[xy[:, 1], xy[:, 0], :3].mean(axis=0)
    if node.type == "RGB":
        return np.array(node.outputs[0].default_value[:3])
    colours = [i for i in node.inputs if i.type == "RGBA" and i.enabled]
    if node.type in ("MIX", "MIX_RGB") and len(colours) >= 2:
        fac_socket = next(i for i in node.inputs if i.type == "VALUE" and i.enabled)
        fac = float(fac_socket.default_value) if not fac_socket.is_linked else 0.5
        a = _socket_rgb(colours[0], uv, cache, depth + 1)
        b = _socket_rgb(colours[1], uv, cache, depth + 1)
        mixed = a * b if node.blend_type == "MULTIPLY" else b
        return (1 - fac) * a + fac * mixed
    if colours:
        return _socket_rgb(colours[0], uv, cache, depth + 1)
    return np.array(socket.default_value[:3])


def _image_array(image, cache: dict) -> np.ndarray:
    if image.name not in cache:
        small = image.copy()
        small.scale(128, 128)
        px = np.empty(128 * 128 * 4, np.float32)
        small.pixels.foreach_get(px)
        bpy.data.images.remove(small)
        cache[image.name] = px.reshape(128, 128, 4)
    return cache[image.name]


def brightness(piece: Piece, cache: dict) -> float:
    """Area-weighted luminance of the base colour, sampling textures at the
    piece's own UVs so texture atlases shared by both colours still work."""
    total = weight = 0.0
    for part in piece.parts:
        mesh = part.obj.data
        n = len(mesh.polygons)
        mat_index, loop_start, area = np.empty(n, int), np.empty(n, int), np.empty(n)
        mesh.polygons.foreach_get("material_index", mat_index)
        mesh.polygons.foreach_get("loop_start", loop_start)
        mesh.polygons.foreach_get("area", area)
        uv = None
        if mesh.uv_layers.active is not None:
            uv = np.empty(len(mesh.loops) * 2)
            mesh.uv_layers.active.data.foreach_get("uv", uv)
            uv = uv.reshape(-1, 2)
        for i, slot in enumerate(part.obj.material_slots):
            sel = mat_index == i
            if not sel.any() or slot.material is None:
                continue
            bsdf = _principled(slot.material)
            if bsdf is None:
                rgb = np.array(slot.material.diffuse_color[:3])
            else:
                rgb = _socket_rgb(
                    bsdf.inputs["Base Color"],
                    uv[loop_start[sel]] if uv is not None else None,
                    cache,
                )
            lum = float(rgb @ np.array([0.2126, 0.7152, 0.0722]))
            w = float(area[sel].sum())
            total, weight = total + lum * w, weight + w
    return total / weight if weight else 0.0


def assign_colours(pieces: list[Piece], swap: bool) -> str:
    """The lighter half is white. Names only decide when the two halves are
    too close in brightness to tell apart, since some models label their
    light-wood pieces "Black"."""
    cache: dict = {}
    lum = np.array([brightness(p, cache) for p in pieces])
    order = np.argsort(lum)
    half = len(pieces) // 2
    lighter = np.zeros(len(pieces), bool)
    lighter[order[half:]] = True
    dark, light = lum[~lighter], lum[lighter]
    gap = (light.min() - dark.max()) / max(light.mean(), 1e-6)

    named = []
    for p in pieces:
        hits = [c for c, pat in COLOUR_PATTERNS.items() if re.search(pat, p.text)]
        named.append(hits[0] if len(hits) == 1 else None)
    names_usable = all(named) and named.count("white") == named.count("black")

    if gap < 0.1 and names_usable:
        method = "names"
        for p, c in zip(pieces, named):
            p.colour = c
    else:
        method = "brightness"
        for p, is_light in zip(pieces, lighter):
            p.colour = "white" if is_light else "black"
    if swap:
        for p in pieces:
            p.colour = "white" if p.colour == "black" else "black"
    return method


BACK_RANK = "rnbqkbnr"


def knight_score(piece: Piece) -> float:
    """High for a knight's head: off the base axis and longer than wide."""
    return piece.head_offset + piece.elongation - 1


def assign_types(pieces: list[Piece]) -> str:
    """Starting layout first: when all 32 pieces stand on the squares of the
    initial position, the squares say what each piece is (king and queen by
    height, in case the model swapped them). Then names, then shape."""
    squares = {p.square: p for p in pieces}
    start = {(f, r): "p" for f in range(8) for r in (1, 6)}
    start.update({(f, r): BACK_RANK[f] for f in range(8) for r in (0, 7)})
    if len(squares) == 32 == len(pieces) and set(squares) == set(start):
        colour_ok = all((p.colour == "white") == (p.square[1] < 2) for p in pieces)
        if colour_ok:
            for p in pieces:
                p.type = start[p.square]
            for rank in (0, 7):
                d, e = squares[(3, rank)], squares[(4, rank)]
                d.type, e.type = ("k", "q") if d.height > e.height else ("q", "k")
            # Some models stand the bishops on the knights' squares, sometimes for
            # one colour only; the pair with the off-axis, elongated heads is the knights
            swapped = []
            for colour in ("white", "black"):
                knights = [p for p in pieces if p.colour == colour and p.type == "n"]
                bishops = [p for p in pieces if p.colour == colour and p.type == "b"]
                if np.mean([knight_score(p) for p in bishops]) > np.mean(
                    [knight_score(p) for p in knights]
                ):
                    for p in knights:
                        p.type = "b"
                    for p in bishops:
                        p.type = "n"
                    swapped.append(colour)
            if swapped:
                return f"start, {' and '.join(swapped)} knights and bishops swapped"
            return "start"

    named = []
    for p in pieces:
        hit = next((t for t, pat in TYPE_PATTERNS if re.search(pat, p.text)), None)
        named.append(hit)
    expected = sorted("ppppppppnnbbrrqk" * 2)
    found = [t for t in named if t is not None]
    if len(found) == len(named) and sorted(found) == expected:
        for p, t in zip(pieces, found):
            p.type = t
        return "names"
    for colour in ("white", "black"):
        side = sorted((p for p in pieces if p.colour == colour), key=lambda p: p.height)
        if len(side) != 16:
            raise SystemExit(f"{len(side)} {colour} pieces; shape-based typing needs all 16")
        for p in side[:8]:
            p.type = "p"
        rest = side[8:]
        rest[-1].type, rest[-2].type = "k", "q"
        minor = sorted(rest[:6], key=lambda p: -knight_score(p))
        for p in minor[:2]:
            p.type = "n"
        rook_bishop = sorted(minor[2:], key=lambda p: p.height)
        for p, t in zip(rook_bishop, "rrbb"):
            p.type = t
    return "shape"


# -- board frame ---------------------------------------------------------------------


def board_angle(board: list[Part]) -> float:
    """Rotation in [-45, 45) degrees that makes the board's footprint an
    axis-aligned square, i.e. minimises its bounding-box area."""
    xy = np.concatenate([vertices(p.obj)[:, :2] for p in board])
    xy = xy[:: max(1, len(xy) // 20000)] - xy.mean(axis=0)
    best = min(
        np.arange(-45, 45, 0.25),
        key=lambda a: np.prod(
            np.ptp(
                xy
                @ np.array(
                    [
                        [math.cos(math.radians(a)), -math.sin(math.radians(a))],
                        [math.sin(math.radians(a)), math.cos(math.radians(a))],
                    ]
                ).T,
                axis=0,
            )
        ),
    )
    return -float(best)


def board_frame(pieces: list[Piece], board_centre: np.ndarray, grid_angle: float) -> dict:
    """Square size and centre from the piece grid; rotation from the board
    outline plus which side the white pieces stand on."""
    bases = np.array([p.base for p in pieces])
    diff = bases[:, None] - bases[None]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest = dist.argmin(axis=1)
    square = float(np.median(dist[np.arange(len(bases)), nearest]))

    # Turn so white sits on -y
    white = bases[[p.colour == "white" for p in pieces]].mean(axis=0)
    black = bases[[p.colour == "black" for p in pieces]].mean(axis=0)
    toward_black = math.degrees(math.atan2(*(black - white)[::-1]))
    turn = 90 * round((90 - (toward_black - grid_angle)) / 90) - grid_angle

    rot = np.array(
        [
            [math.cos(math.radians(turn)), -math.sin(math.radians(turn))],
            [math.sin(math.radians(turn)), math.cos(math.radians(turn))],
        ]
    )
    turned = bases @ rot.T
    # Square centres sit at centre + (i - 3.5) * square, so the grid phase fixes the centre
    # up to whole squares, and the board's own bounding box picks which one
    phase = np.angle(np.exp(2j * np.pi * (turned / square - 0.5)).mean(axis=0)) / (2 * np.pi)
    guess = rot @ board_centre / square
    centre = square * (phase + np.round(guess - phase))
    board_centre_rotated = rot @ board_centre
    residual = turned - centre
    off_grid = np.abs(residual / square - np.round(residual / square - 0.5) - 0.5).max()
    return {
        "square": square,
        "turn": turn,
        "centre_rotated": centre,
        "grid_vs_board": [float(v) for v in (centre - board_centre_rotated) / square],
        "max_off_grid": float(off_grid),
        "rotation": rot,
    }


def piece_template(piece: Piece, name: str, transform: Matrix) -> bpy.types.Object:
    obj = join([p.obj for p in piece.parts], name)
    obj.data.transform(transform)
    v = vertices(obj)
    lo = v.min(axis=0)
    height = v[:, 2].max() - lo[2]
    base = v[v[:, 2] < lo[2] + 0.1 * height]
    centre = (base.min(axis=0) + base.max(axis=0)) / 2
    obj.data.transform(Matrix.Translation((-centre[0], -centre[1], -lo[2])))

    # Some models ship inward-facing normals that a double-sided viewer hides
    # but Cycles shades dark, so rebuild them pointing out
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    with bpy.context.temp_override(object=obj, active_object=obj):
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
    obj.data.set_sharp_from_angle(angle=math.radians(40))

    if piece.type == "n":
        # Point the snout, the head vertex furthest from the base axis, along +y
        v = vertices(obj)
        top = v[v[:, 2] > 0.6 * height]
        tip = top[np.argmax(np.linalg.norm(top[:, :2], axis=1))]
        obj.data.transform(Matrix.Rotation(math.pi / 2 - math.atan2(tip[1], tip[0]), 4, "Z"))
    obj.hide_render = True
    return obj


def board_brightness(board: bpy.types.Object, x: float, y: float, cache: dict) -> float | None:
    """Luminance of the board's base colour straight below (x, y)."""
    bpy.context.view_layer.update()
    hit, location, _, index = board.ray_cast(Vector((x, y, 10.)), Vector((0, 0, -1)))
    if not hit:
        return None
    mesh = board.data
    poly = mesh.polygons[index]
    slot = board.material_slots[poly.material_index] if board.material_slots else None
    if slot is None or slot.material is None:
        return None
    bsdf = _principled(slot.material)
    if bsdf is None:
        rgb = np.array(slot.material.diffuse_color[:3])
    else:
        uv = None
        if mesh.uv_layers.active is not None:
            weights = poly_3d_calc([mesh.vertices[v].co for v in poly.vertices], location)
            uvs = [mesh.uv_layers.active.data[i].uv for i in poly.loop_indices]
            uv = np.array([sum((w * u for w, u in zip(weights, uvs)), Vector((0, 0)))])
        rgb = _socket_rgb(bsdf.inputs["Base Color"], uv, cache)
    return float(rgb @ np.array([0.2126, 0.7152, 0.0722]))


def fit_checker(board: bpy.types.Object, square: float) -> dict:
    """Fit a checker pattern to the board's own colours.

    Brightness is sampled on a grid over the playing area and matched against
    +-1 checker patterns over square sizes within 15% of ``square`` and all
    offsets. Returns the best square size, the centre (the board's middle
    corner, nearest the current origin), whether a1 is dark, and a confidence
    in [0, 1] (1 = a perfect two-tone checker).
    """
    cache: dict = {}
    xs = np.linspace(-3.8, 3.8, 96) * square
    lum = np.array(
        [[board_brightness(board, x, y, cache) for x in xs] for y in xs], dtype=float
    )
    valid = ~np.isnan(lum)
    if valid.mean() < 0.5:
        return {"confidence": 0.0}
    lum = np.where(valid, lum - np.nanmean(lum), 0.0)
    total = np.abs(lum).sum()
    if total == 0:
        return {"confidence": 0.0}

    best = (0.0, square, 0.0, 0.0, 0.0)
    for size in square * np.linspace(0.85, 1.18, 67):
        phases = np.linspace(0, size, 32, endpoint=False)
        pattern = np.sign(np.sin(np.pi * (xs[None, :] - phases[:, None]) / size))
        corr = pattern @ lum @ pattern.T  # [y phase, x phase]
        iy, ix = np.unravel_index(np.abs(corr).argmax(), corr.shape)
        if abs(corr[iy, ix]) / total > best[0]:
            best = (abs(corr[iy, ix]) / total, size, phases[ix], phases[iy], corr[iy, ix])
    confidence, size, cx, cy, corr = best

    # Square edges sit at c + k * size; the board's middle is the edge nearest
    # the origin. Each whole-square shift flips which squares the pattern calls dark
    kx, ky = round(cx / size), round(cy / size)
    cx, cy = cx - kx * size, cy - ky * size
    sign = corr * (-1) ** (kx + ky)
    # The square up and right of the middle is e5, a dark square, where the
    # pattern is +1; a negative correlation means it really is dark
    return {
        "confidence": float(confidence),
        "square": float(size),
        "centre": (float(cx), float(cy)),
        "a1_dark": bool(sign < 0),
    }


def render_preview(path: Path, templates: dict, square: float, device: str) -> None:
    scene = bpy.context.scene
    setup_device(scene, device)
    scene.render.resolution_x, scene.render.resolution_y = 1000, 750
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    copies = []
    for symbol, file, rank in parse_position(START).pieces():
        obj = templates[symbol].copy()
        obj.hide_render = False
        obj.location = ((file - 3.5) * square, (rank - 3.5) * square, 0)
        scene.collection.objects.link(obj)
        copies.append(obj)
    camera = bpy.data.objects.new("preview_camera", bpy.data.cameras.new("preview_camera"))
    scene.collection.objects.link(camera)
    camera.data.lens = 35
    camera.matrix_world = look_at(Vector((0, -12 * square, 11 * square)), Vector((0, 0, 0)))
    sun = bpy.data.objects.new("preview_sun", bpy.data.lights.new("preview_sun", "SUN"))
    scene.collection.objects.link(sun)
    sun.data.energy = 3
    sun.matrix_world = look_at(Vector((1, -1, 2)), Vector((0, 0, 0)))
    scene.world = scene.world or bpy.data.worlds.new("world")
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.4
    scene.camera = camera
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    for obj in [*copies, camera, sun]:
        bpy.data.objects.remove(obj, do_unlink=True)
    scene.camera = None


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = parse_args(argv)
    load_model(args.model)
    parts = [Part(o) for o in separate_loose(bake_world_transforms())]
    pieces, scenery, rough, board_top = find_pieces(parts)
    warnings = []
    if len(pieces) != 32:
        warnings.append(f"found {len(pieces)} pieces, expected 32")

    colour_method = assign_colours(pieces, args.swap_colours)
    if args.verbose:
        for p in sorted(pieces, key=lambda p: (p.colour, p.height)):
            print(
                f"{p.colour:5s} h={p.height:8.3f} elong={p.elongation:5.2f} "
                f"head={p.head_offset:5.2f} "
                f"base=({p.base[0]:8.3f},{p.base[1]:8.3f}) parts={len(p.parts):3d} {p.text[:70]}"
            )
    # The board can be several large parts (border strips, a base, the squares)
    board_parts = [p for p in scenery if p.footprint > 2 * rough] or [
        max(scenery, key=lambda p: p.footprint)
    ]
    lo = np.min([p.lo[:2] for p in board_parts], axis=0)
    hi = np.max([p.hi[:2] for p in board_parts], axis=0)
    frame = board_frame(pieces, (lo + hi) / 2, board_angle(board_parts))
    if frame["max_off_grid"] > 0.15:
        warnings.append(f"a piece sits {frame['max_off_grid']:.2f} squares off the grid")
    for p in pieces:
        x, y = frame["rotation"] @ p.base - frame["centre_rotated"]
        p.square = (int(round(x / frame["square"] + 3.5)), int(round(y / frame["square"] + 3.5)))
    type_method = assign_types(pieces)

    scale = args.square_size / frame["square"]
    cx, cy = frame["centre_rotated"]
    transform = (
        Matrix.Scale(scale, 4)
        @ Matrix.Rotation(math.radians(args.rotate), 4, "Z")
        @ Matrix.Translation((-cx, -cy, -board_top))
        @ Matrix.Rotation(math.radians(frame["turn"]), 4, "Z")
    )

    counts: dict[str, int] = {}
    chosen = {}
    for piece in pieces:
        assert piece.type is not None  # assign_types types every piece or exits
        symbol = piece.type.upper() if piece.colour == "white" else piece.type
        counts[symbol] = counts.get(symbol, 0) + 1
        chosen.setdefault(symbol, piece)
    missing = [s for s in "PNBRQKpnbrqk" if s not in chosen]
    if missing:
        raise SystemExit(f"no piece found for {missing}")
    for piece in pieces:
        if piece not in chosen.values():
            for part in piece.parts:
                bpy.data.objects.remove(part.obj, do_unlink=True)
    templates = {}
    for symbol, name in PIECE_NAMES.items():
        for colour, sym in (("White", symbol.upper()), ("Black", symbol)):
            templates[sym] = piece_template(chosen[sym], f"{colour} {name}", transform)

    # Anything left floating above the board surface is debris, not board
    keep = [p.obj for p in scenery if p.lo[2] <= board_top + 0.05 * rough]
    for p in scenery:
        if p.obj not in keep:
            bpy.data.objects.remove(p.obj, do_unlink=True)
    board = join(keep, "board")
    board.data.transform(transform)

    # Some boards' squares do not match the piece grid exactly, or sit a
    # quarter turn off it (a dark square at h1). When the board has a clear
    # checker pattern, its own squares set the scale, centre and turn
    fit = fit_checker(board, args.square_size)
    board_turn = args.board_turn
    if fit["confidence"] >= 0.6:
        cx, cy = fit["centre"]
        k = args.square_size / fit["square"]
        board.data.transform(Matrix.Scale(k, 4) @ Matrix.Translation((-cx, -cy, 0)))
        for obj in templates.values():
            obj.data.transform(Matrix.Scale(k, 4))
        if board_turn is None:
            board_turn = 0.0 if fit["a1_dark"] else 90.0
    else:
        warnings.append(
            f"no clear checker pattern on the board (fit {fit['confidence']:.2f}); "
            "check that h1 is light"
        )
    if board_turn:
        board.data.transform(Matrix.Rotation(math.radians(board_turn), 4, "Z"))

    expected = {"P": 8, "N": 2, "B": 2, "R": 2, "Q": 1, "K": 1}
    for sym, n in expected.items():
        for s in (sym, sym.lower()):
            if counts.get(s, 0) != n:
                warnings.append(f"{counts.get(s, 0)} '{s}' pieces, expected {n}")

    scene = bpy.context.scene
    scene["square_size"] = args.square_size
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.preview:
        render_preview(
            args.out.with_suffix(".preview.png"), templates, args.square_size, args.device
        )
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out.resolve()), compress=True)

    report = {
        "model": str(args.model),
        "pieces": len(pieces),
        "counts": counts,
        "colour_method": colour_method,
        "type_method": type_method,
        "model_square_size": frame["square"],
        "scale": scale,
        "rotation_deg": frame["turn"] + args.rotate,
        "max_off_grid": frame["max_off_grid"],
        "grid_vs_board": frame["grid_vs_board"],
        "board_turn_deg": board_turn or 0.0,
        "checker_fit": {k: v for k, v in fit.items()},
        "warnings": warnings,
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

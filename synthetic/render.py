"""Render synthetic chessboard photos with Blender Cycles.

Each line of a FEN list becomes one JPEG plus a JSON label in ChessCog's
layout (``<out>/render/<split>/<id>.{jpg,json}``), so the renders load with
`chessqueries.data.chesscog.ChessCog(dataroot=<out>)`. Labels carry the FEN,
side to move, board corner pixels, per-piece boxes, and every sampled
camera/lighting/material parameter.

Each FEN is rendered on a chess set picked at random from ``--sets``: the
``scene.blend`` files that ``prepare_set.py`` writes under
``data/chess_sets/<set>/``. Each file is laid out like ChessCog's scene:
board centred on the origin with its top at z=0, white on the -y side, a
``square_size`` property, and piece templates named "White Pawn" ...
"Black King" with their origins at the base of the piece. The pick depends
only on ``--seed`` and the FEN's line number, so reruns and shards agree;
FENs are grouped by set so each .blend is opened once per run.
``--procedural`` uses the built-in scene instead: a table, a board in a
random colour scheme, and lathe-turned Staunton-like pieces.

    blender -b -P synthetic/render.py -- --fens data/synthetic/fens.txt --out data/synthetic

Written for Blender 5. It also runs under the matching ``bpy`` wheel from PyPI
(Python 3.13):
``python synthetic/render.py --fens ...``. Rendering uses the GPU and stops
if Cycles finds none; ``--device CPU`` renders on the CPU on purpose.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import bpy  # first: the PyPI wheel only exposes bmesh and mathutils once bpy is imported
import bmesh
import numpy as np
from mathutils import Euler, Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from positions import assign_split, read_positions, square_name  # noqa: E402

PIECE_NAMES = {"p": "Pawn", "n": "Knight", "b": "Bishop", "r": "Rook", "q": "Queen", "k": "King"}


# Lathe profiles as (radius, height) in square widths, bottom to top
def _arc(cz, r, start_deg, end_deg, steps=8):
    angles = np.linspace(math.radians(start_deg), math.radians(end_deg), steps)
    return [(r * math.cos(a), cz + r * math.sin(a)) for a in angles]


PROFILES = {
    "p": [
        (0, 0),
        (0.30, 0),
        (0.30, 0.05),
        (0.27, 0.09),
        (0.21, 0.12),
        (0.19, 0.16),
        (0.13, 0.21),
        (0.10, 0.46),
        (0.17, 0.50),
        (0.17, 0.53),
        (0.09, 0.56),
    ]
    + _arc(0.70, 0.14, -60, 90),
    "r": [
        (0, 0),
        (0.34, 0),
        (0.34, 0.06),
        (0.30, 0.10),
        (0.25, 0.13),
        (0.22, 0.18),
        (0.19, 0.30),
        (0.18, 0.62),
        (0.24, 0.68),
        (0.24, 0.72),
        (0.22, 0.74),
        (0.22, 0.92),
        (0, 0.92),
    ],
    "n": [
        (0, 0),
        (0.34, 0),
        (0.34, 0.06),
        (0.30, 0.10),
        (0.25, 0.13),
        (0.24, 0.20),
        (0.20, 0.24),
        (0, 0.24),
    ],
    "b": [
        (0, 0),
        (0.32, 0),
        (0.32, 0.06),
        (0.28, 0.10),
        (0.22, 0.13),
        (0.20, 0.18),
        (0.13, 0.26),
        (0.10, 0.66),
        (0.18, 0.70),
        (0.18, 0.73),
        (0.10, 0.76),
        (0.15, 0.82),
        (0.17, 0.90),
        (0.16, 0.98),
        (0.12, 1.06),
        (0.06, 1.12),
        (0.03, 1.15),
    ]
    + _arc(1.20, 0.055, -60, 90),
    "q": [
        (0, 0),
        (0.36, 0),
        (0.36, 0.06),
        (0.32, 0.10),
        (0.25, 0.14),
        (0.22, 0.20),
        (0.14, 0.30),
        (0.11, 0.86),
        (0.20, 0.90),
        (0.20, 0.94),
        (0.11, 0.98),
        (0.13, 1.04),
        (0.20, 1.18),
        (0.22, 1.24),
        (0.15, 1.26),
        (0.10, 1.30),
    ]
    + _arc(1.36, 0.07, -45, 90),
    "k": [
        (0, 0),
        (0.37, 0),
        (0.37, 0.06),
        (0.33, 0.10),
        (0.26, 0.14),
        (0.23, 0.20),
        (0.15, 0.30),
        (0.12, 0.96),
        (0.21, 1.00),
        (0.21, 1.04),
        (0.12, 1.08),
        (0.15, 1.14),
        (0.21, 1.30),
        (0.20, 1.36),
        (0.08, 1.40),
        (0, 1.40),
    ],
}
# Side view of the knight's head as (y, z); the snout points along +y
KNIGHT_HEAD = [
    (-0.22, 0.22),
    (0.22, 0.22),
    (0.18, 0.40),
    (0.30, 0.62),
    (0.38, 0.74),
    (0.36, 0.84),
    (0.22, 0.90),
    (0.10, 0.98),
    (0.04, 1.12),
    (-0.04, 1.06),
    (-0.12, 1.02),
    (-0.22, 0.86),
    (-0.28, 0.62),
    (-0.26, 0.40),
]
KNIGHT_HEAD_WIDTH = 0.26

# (dark square, light square) in sRGB
BOARD_PALETTES = {
    "tournament_green": ((0.46, 0.59, 0.34), (0.93, 0.93, 0.82)),
    "wood_brown": ((0.55, 0.36, 0.22), (0.87, 0.73, 0.53)),
    "walnut": ((0.35, 0.22, 0.14), (0.78, 0.62, 0.42)),
    "blue_vinyl": ((0.33, 0.45, 0.60), (0.90, 0.91, 0.88)),
    "black_white": ((0.15, 0.15, 0.15), (0.88, 0.88, 0.85)),
    "grey": ((0.45, 0.45, 0.45), (0.85, 0.85, 0.85)),
}
WHITE_PIECE_COLOURS = [
    (0.93, 0.90, 0.80),
    (0.90, 0.78, 0.58),
    (0.95, 0.95, 0.93),
    (0.85, 0.82, 0.74),
]
BLACK_PIECE_COLOURS = [
    (0.05, 0.05, 0.05),
    (0.22, 0.12, 0.07),
    (0.10, 0.08, 0.07),
    (0.16, 0.16, 0.17),
]
TABLE_COLOURS = [
    (0.45, 0.30, 0.18),
    (0.70, 0.55, 0.38),
    (0.25, 0.17, 0.11),
    (0.80, 0.80, 0.78),
    (0.30, 0.30, 0.32),
    (0.60, 0.45, 0.35),
]


def srgb_to_linear(rgb):
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb)


def jitter_colour(rng: random.Random, rgb, amount=0.08):
    brightness = rng.uniform(1 - amount, 1 + amount)
    return tuple(
        min(1.0, max(0.0, c * brightness * rng.uniform(1 - amount / 2, 1 + amount / 2)))
        for c in rgb
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render synthetic chessboard photos.")
    parser.add_argument(
        "--fens",
        type=Path,
        required=True,
        help="FEN list: full FENs, bare placements, or ChessCog's W/B-prefixed lines",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="dataset root to write render/ into"
    )
    parser.add_argument("--start", type=int, default=0, help="first FEN line to render (sharding)")
    parser.add_argument("--end", type=int, default=None, help="stop before this FEN line")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", default="1200x800", help="WIDTHxHEIGHT in pixels")
    parser.add_argument(
        "--samples", type=int, default=64, help="Cycles samples per pixel (denoised)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (first GPU backend found), OPTIX, CUDA, HIP, ONEAPI, METAL, or CPU",
    )
    parser.add_argument(
        "--splits",
        default="train=0.9,val=0.03,test=0.07",
        help="split fractions, assigned by hashing the placement",
    )
    parser.add_argument(
        "--sets",
        type=Path,
        nargs="+",
        default=[Path("data/chess_sets")],
        help="scene.blend files, or directories searched for */scene.blend, to pick from per FEN",
    )
    parser.add_argument(
        "--procedural",
        action="store_true",
        help="render every FEN on the built-in procedural scene instead of --sets",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=None,
        help="square width in metres (default: each set's square_size property, else 0.05)",
    )
    parser.add_argument(
        "--angle-min", type=float, default=35.0, help="min camera elevation, degrees"
    )
    parser.add_argument(
        "--angle-max", type=float, default=75.0, help="max camera elevation, degrees"
    )
    parser.add_argument(
        "--yaw-sigma",
        type=float,
        default=15.0,
        help="std of the camera's azimuth around the mover's side, degrees",
    )
    parser.add_argument(
        "--any-side-prob",
        type=float,
        default=0.15,
        help="probability of a uniformly random azimuth instead",
    )
    parser.add_argument(
        "--focal-min", type=float, default=24.0, help="min focal length, mm (36 mm sensor)"
    )
    parser.add_argument("--focal-max", type=float, default=60.0, help="max focal length, mm")
    parser.add_argument("--dof-prob", type=float, default=0.3, help="probability of depth of field")
    parser.add_argument(
        "--piece-jitter",
        type=float,
        default=0.08,
        help="std of a piece's offset from its square centre, in square widths",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=16,
        help="minimum distance in pixels from board corners and pieces to the frame edge",
    )
    args = parser.parse_args(argv)
    args.width, args.height = (int(v) for v in args.resolution.lower().split("x"))
    args.split_fractions = {
        k: float(v) for k, v in (item.split("=") for item in args.splits.split(","))
    }
    if abs(sum(args.split_fractions.values()) - 1) > 1e-6:
        parser.error("--splits fractions must sum to 1")
    return args


# -- scene construction ---------------------------------------------------------


def make_material(name: str) -> tuple[bpy.types.Material, bpy.types.Node]:
    mat = bpy.data.materials.new(name)
    return mat, mat.node_tree.nodes["Principled BSDF"]


def make_wood_material(name: str):
    """Principled BSDF coloured by stretched noise, giving a grain-like pattern."""
    mat, bsdf = make_material(name)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    coords = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    noise = nodes.new("ShaderNodeTexNoise")
    ramp = nodes.new("ShaderNodeValToRGB")
    links.new(coords.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    noise.inputs["Detail"].default_value = 6.0
    return mat, bsdf, mapping, noise, ramp


def build_piece_mesh(symbol: str, square_size: float) -> bpy.types.Mesh:
    bm = bmesh.new()
    verts = [bm.verts.new((r, 0.0, z)) for r, z in PROFILES[symbol]]
    edges = [bm.edges.new(pair) for pair in zip(verts, verts[1:])]
    bmesh.ops.spin(
        bm,
        geom=verts + edges,
        cent=(0, 0, 0),
        axis=(0, 0, 1),
        angle=2 * math.pi,
        steps=48,
        use_merge=True,
    )
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=bm.edges)

    if symbol == "n":
        half = KNIGHT_HEAD_WIDTH / 2
        left = [bm.verts.new((-half, y, z)) for y, z in KNIGHT_HEAD]
        right = [bm.verts.new((half, y, z)) for y, z in KNIGHT_HEAD]
        bm.faces.new(left)
        bm.faces.new(list(reversed(right)))
        n = len(KNIGHT_HEAD)
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((left[j], left[i], right[i], right[j]))
    elif symbol == "r":
        for i in range(6):
            a = i * math.pi / 3
            bmesh.ops.create_cube(
                bm,
                size=1.0,
                matrix=Matrix.LocRotScale(
                    (0.18 * math.cos(a), 0.18 * math.sin(a), 0.97),
                    Euler((0, 0, a)),
                    (0.09, 0.10, 0.10),
                ),
            )
    elif symbol == "q":
        for i in range(8):
            a = i * math.pi / 4
            bmesh.ops.create_uvsphere(
                bm,
                u_segments=12,
                v_segments=8,
                radius=0.04,
                matrix=Matrix.Translation((0.21 * math.cos(a), 0.21 * math.sin(a), 1.26)),
            )
    elif symbol == "k":
        bmesh.ops.create_cube(
            bm, size=1.0, matrix=Matrix.LocRotScale((0, 0, 1.52), None, (0.06, 0.06, 0.24))
        )
        bmesh.ops.create_cube(
            bm, size=1.0, matrix=Matrix.LocRotScale((0, 0, 1.56), None, (0.18, 0.06, 0.06))
        )

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.scale(bm, vec=(square_size,) * 3, verts=bm.verts)
    for face in bm.faces:
        face.smooth = True
    mesh = bpy.data.meshes.new(f"piece_{symbol}")
    bm.to_mesh(mesh)
    bm.free()
    mesh.set_sharp_from_angle(angle=math.radians(40))
    return mesh


class ProceduralScene:
    """Table, board and pieces built from code, with per-render random looks."""

    def __init__(self, scene: bpy.types.Scene, square_size: float) -> None:
        self.scene = scene
        self.square_size = s = square_size
        self.board_mats = {k: make_material(f"square_{k}") for k in ("dark", "light")}
        self.frame = make_wood_material("board_frame")
        self.table = make_wood_material("table")
        self.piece_mats = {c: make_material(f"pieces_{c}") for c in ("white", "black")}

        # Squares: one quad each, a1 (dark) at (-4s, -4s)
        verts, faces = [], []
        for rank in range(8):
            for file in range(8):
                x0, y0 = (file - 4) * s, (rank - 4) * s
                base = len(verts)
                verts += [(x0, y0, 0), (x0 + s, y0, 0), (x0 + s, y0 + s, 0), (x0, y0 + s, 0)]
                faces.append((base, base + 1, base + 2, base + 3))
        mesh = bpy.data.meshes.new("squares")
        mesh.from_pydata(verts, [], faces)
        mesh.materials.append(self.board_mats["dark"][0])
        mesh.materials.append(self.board_mats["light"][0])
        for i, poly in enumerate(mesh.polygons):
            file, rank = i % 8, i // 8
            poly.material_index = 0 if (file + rank) % 2 == 0 else 1
        self._link(bpy.data.objects.new("squares", mesh))

        # Frame: a slab just below the squares, resized per render for the border width
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        frame_mesh = bpy.data.meshes.new("frame")
        bm.to_mesh(frame_mesh)
        bm.free()
        frame_mesh.materials.append(self.frame[0])
        self.frame_obj = self._link(bpy.data.objects.new("frame", frame_mesh))

        bm = bmesh.new()
        bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=60 * s)
        table_mesh = bpy.data.meshes.new("table")
        bm.to_mesh(table_mesh)
        bm.free()
        table_mesh.materials.append(self.table[0])
        self.table_obj = self._link(bpy.data.objects.new("table", table_mesh))

        # Templates are never linked to the scene, so only their copies render
        self.templates = {}
        for symbol in PROFILES:
            mesh = build_piece_mesh(symbol, s)
            for colour, sym in (("white", symbol.upper()), ("black", symbol)):
                coloured = mesh.copy()
                coloured.materials.append(self.piece_mats[colour][0])
                self.templates[sym] = bpy.data.objects.new(sym, coloured)

    def _link(self, obj):
        self.scene.collection.objects.link(obj)
        return obj

    def randomise(self, rng: random.Random) -> dict:
        s = self.square_size
        palette = rng.choice(sorted(BOARD_PALETTES))
        for key, rgb in zip(("dark", "light"), BOARD_PALETTES[palette]):
            bsdf = self.board_mats[key][1]
            bsdf.inputs["Base Color"].default_value = (*srgb_to_linear(jitter_colour(rng, rgb)), 1)
            bsdf.inputs["Roughness"].default_value = rng.uniform(0.25, 0.8)

        border = rng.uniform(0.25, 0.9) * s
        thickness = rng.uniform(0.15, 0.5) * s
        half = 4 * s + border
        self.frame_obj.scale = (2 * half, 2 * half, thickness)
        self.frame_obj.location = (0, 0, -thickness / 2 - 1e-4)
        self.table_obj.location = (rng.gauss(0, 2 * s), rng.gauss(0, 2 * s), -thickness - 1e-4)
        self.table_obj.rotation_euler = (0, 0, rng.uniform(0, 2 * math.pi))
        for target, colours, grain in (
            (self.frame, BOARD_PALETTES[palette][0], 30.0),
            (self.table, rng.choice(TABLE_COLOURS), 3.0),
        ):
            _, bsdf, mapping, noise, ramp = target
            base = jitter_colour(rng, colours, 0.15)
            darker = tuple(c * rng.uniform(0.55, 0.85) for c in base)
            ramp.color_ramp.elements[0].color = (*srgb_to_linear(darker), 1)
            ramp.color_ramp.elements[1].color = (*srgb_to_linear(base), 1)
            # Stretch the noise along one axis for grain; scale 1 is near-uniform
            stretch = rng.choice([1.0, rng.uniform(8, 40)])
            mapping.inputs["Scale"].default_value = (grain, grain * stretch, grain)
            noise.inputs["Scale"].default_value = rng.uniform(2, 8)
            bsdf.inputs["Roughness"].default_value = rng.uniform(0.3, 0.85)

        looks = {}
        for colour, choices in (("white", WHITE_PIECE_COLOURS), ("black", BLACK_PIECE_COLOURS)):
            rgb = jitter_colour(rng, rng.choice(choices))
            bsdf = self.piece_mats[colour][1]
            bsdf.inputs["Base Color"].default_value = (*srgb_to_linear(rgb), 1)
            bsdf.inputs["Roughness"].default_value = rng.uniform(0.15, 0.6)
            looks[colour] = [round(c, 3) for c in rgb]
        return {"board_palette": palette, "border": border / s, "pieces": looks}


class LoadedScene:
    """A user .blend with ChessCog-style piece templates; looks are left as authored."""

    def __init__(self, scene: bpy.types.Scene, square_size: float) -> None:
        self.square_size = square_size
        self.templates = {}
        for symbol, name in PIECE_NAMES.items():
            for colour, sym in (("White", symbol.upper()), ("Black", symbol)):
                obj = bpy.data.objects.get(f"{colour} {name}")
                if obj is None:
                    raise SystemExit(f"the set is missing the piece template {colour} {name!r}")
                obj.hide_render = True
                self.templates[sym] = obj

    def randomise(self, rng: random.Random) -> dict:
        return {}


# -- camera, lights, rendering --------------------------------------------------


def look_at(location: Vector, target: Vector, roll: float = 0.0) -> Matrix:
    rot = (target - location).to_track_quat("-Z", "Y").to_matrix().to_4x4()
    return Matrix.Translation(location) @ rot @ Matrix.Rotation(roll, 4, "Z")


class LightRig:
    def __init__(self, scene: bpy.types.Scene) -> None:
        def add(name, kind):
            obj = bpy.data.objects.new(name, bpy.data.lights.new(name, kind))
            scene.collection.objects.link(obj)
            return obj

        self.flash = add("flash", "SPOT")
        self.spots = [add("spot1", "SPOT"), add("spot2", "SPOT")]
        self.area = add("overhead", "AREA")
        self.sun = add("sun", "SUN")
        self.all = [self.flash, *self.spots, self.area, self.sun]

        world = scene.world or bpy.data.worlds.new("world")
        scene.world = world
        self.background = world.node_tree.nodes["Background"]

    def randomise(self, rng: random.Random, square_size: float, camera: Matrix) -> dict:
        """Brightness is set from each light's distance so that, before the
        exposure jitter, every mode lands near the same irradiance."""
        s = square_size
        irradiance = rng.uniform(1.5, 4.5)
        warm, cool = Vector((1.0, 0.82, 0.65)), Vector((0.82, 0.9, 1.0))
        tint = warm.lerp(cool, rng.random())
        for obj in self.all:
            obj.hide_render = True
            obj.data.color = tint

        mode = rng.choice(["flash", "spotlights", "overhead", "sun"])
        info: dict = {"mode": mode, "tint": [round(c, 3) for c in tint]}
        if mode == "flash":
            loc = camera.translation
            self.flash.matrix_world = look_at(loc, Vector((0, 0, 0)))
            self.flash.data.spot_size = math.radians(100)
            self.flash.data.spot_blend = 0.6
            self.flash.data.shadow_soft_size = 0.005
            self.flash.data.energy = irradiance * 4 * math.pi * loc.length**2
            self.flash.hide_render = False
            info["location"] = list(loc)
        elif mode == "spotlights":
            info["spots"] = []
            for spot in self.spots:
                a = rng.uniform(0, 2 * math.pi)
                radius = rng.uniform(6, 14) * s
                loc = Vector((radius * math.cos(a), radius * math.sin(a), rng.uniform(5, 14) * s))
                focus = Vector((rng.gauss(0, 1.6 * s), rng.gauss(0, 1.6 * s), 0))
                spot.matrix_world = look_at(loc, focus)
                spot.data.spot_size = math.radians(rng.uniform(40, 90))
                spot.data.spot_blend = rng.uniform(0.2, 1.0)
                spot.data.shadow_soft_size = rng.uniform(0.01, 0.1)
                spot.data.energy = irradiance / 2 * 4 * math.pi * (loc - focus).length ** 2
                spot.hide_render = False
                info["spots"].append({"location": list(loc), "focus": list(focus)})
        elif mode == "overhead":
            size = rng.uniform(4, 12) * s
            loc = Vector((rng.gauss(0, 3 * s), rng.gauss(0, 3 * s), rng.uniform(12, 25) * s))
            self.area.matrix_world = look_at(loc, Vector((0, 0, 0)))
            self.area.data.size = size
            self.area.data.energy = irradiance * math.pi * loc.length**2
            self.area.hide_render = False
            info.update(location=list(loc), size=size)
        else:
            elevation = rng.uniform(15, 70)
            azimuth = rng.uniform(0, 360)
            direction = Vector(
                (
                    math.cos(math.radians(azimuth)) * math.cos(math.radians(elevation)),
                    math.sin(math.radians(azimuth)) * math.cos(math.radians(elevation)),
                    math.sin(math.radians(elevation)),
                )
            )
            self.sun.matrix_world = look_at(direction, Vector((0, 0, 0)))
            self.sun.data.energy = irradiance
            self.sun.data.angle = math.radians(rng.uniform(0.5, 10))
            self.sun.hide_render = False
            info.update(elevation=elevation, azimuth=azimuth)

        ambient = rng.uniform(0.05, 0.45)
        sky = jitter_colour(rng, (0.8, 0.8, 0.8), 0.25)
        self.background.inputs["Color"].default_value = (*srgb_to_linear(sky), 1)
        self.background.inputs["Strength"].default_value = ambient
        info["ambient"] = ambient
        return info


def setup_device(scene: bpy.types.Scene, device: str) -> str:
    scene.render.engine = "CYCLES"
    if device.upper() == "CPU":
        scene.cycles.device = "CPU"
        return "CPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    backends = ["OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"] if device == "auto" else [device.upper()]
    for backend in backends:
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == backend]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == backend
            scene.cycles.device = "GPU"
            return f"{backend}: " + ", ".join(d.name for d in gpus)
    raise SystemExit(f"Cycles found no GPU for {backends}; pass --device CPU to render on the CPU")


def setup_render(scene: bpy.types.Scene, args: argparse.Namespace) -> None:
    render = scene.render
    render.resolution_x, render.resolution_y = args.width, args.height
    render.resolution_percentage = 100
    # Quality-95 JPEG: a fifth the size of PNG and much faster to decode, with no visible loss
    render.image_settings.file_format = "JPEG"
    render.image_settings.quality = 95
    render.image_settings.color_mode = "RGB"
    render.film_transparent = False
    # Keep kernels, textures and geometry on the GPU between renders
    render.use_persistent_data = True
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.denoising_use_gpu = args.device.upper() != "CPU"
    scene.cycles.max_bounces = 6
    scene.view_settings.view_transform = "AgX"


class Projector:
    """World points to pixel coordinates for the scene camera."""

    def __init__(self, scene: bpy.types.Scene, camera: bpy.types.Object) -> None:
        w, h = scene.render.resolution_x, scene.render.resolution_y
        depsgraph = bpy.context.evaluated_depsgraph_get()
        proj = np.array(camera.calc_matrix_camera(depsgraph, x=w, y=h))
        self.matrix = proj @ np.array(camera.matrix_world.inverted())
        self.size = np.array([w, h])

    def __call__(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        clip = np.c_[points, np.ones(len(points))] @ self.matrix.T
        w = clip[:, 3]
        ndc = clip[:, :2] / w[:, None]
        pixels = np.stack([(ndc[:, 0] + 1) / 2, (1 - ndc[:, 1]) / 2], axis=1) * self.size
        return pixels, w > 0

    def inside(self, points: np.ndarray, padding: float) -> bool:
        pixels, in_front = self(points)
        return bool(
            in_front.all() and (pixels >= padding).all() and (pixels <= self.size - padding).all()
        )


def place_camera(
    rng: random.Random,
    args: argparse.Namespace,
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    white_turn: bool,
    must_see: np.ndarray,
) -> dict:
    """Sample a phone-like viewpoint from the mover's side until the board and
    every piece are in frame."""
    s = args.square_size
    sensor = 36.0
    camera.data.sensor_width = sensor
    camera.data.sensor_fit = "AUTO"
    aspect = min(args.width, args.height) / max(args.width, args.height)
    for _ in range(200):
        focal = rng.uniform(args.focal_min, args.focal_max)
        half_fov = math.atan(sensor / 2 * aspect / focal)
        elevation = rng.uniform(args.angle_min, args.angle_max)
        if rng.random() < args.any_side_prob:
            azimuth = rng.uniform(0, 360)
        else:
            azimuth = (-90 if white_turn else 90) + rng.gauss(0, args.yaw_sigma)
        # Distance at which a sphere round the board just fits, then moved closer or further
        distance = 4 * math.sqrt(2) * s / math.sin(half_fov) * rng.uniform(0.75, 1.25)
        target = Vector((rng.gauss(0, 0.3 * s), rng.gauss(0, 0.3 * s), 0))
        el, az = math.radians(elevation), math.radians(azimuth)
        location = target + distance * Vector(
            (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
        )
        roll = math.radians(rng.gauss(0, 3))
        camera.data.lens = focal
        camera.matrix_world = look_at(location, target, roll)
        if Projector(scene, camera).inside(must_see, args.padding):
            break
    else:
        raise RuntimeError("no camera pose kept the board in frame after 200 tries")

    info = {
        "angle": elevation,
        "azimuth": azimuth,
        "distance": distance,
        "focal_length": focal,
        "roll": math.degrees(roll),
        "location": list(location),
        "target": list(target),
        "dof_fstop": None,
    }
    camera.data.dof.use_dof = rng.random() < args.dof_prob
    if camera.data.dof.use_dof:
        camera.data.dof.focus_distance = (location - target).length
        camera.data.dof.aperture_fstop = rng.uniform(2.0, 8.0)
        info["dof_fstop"] = camera.data.dof.aperture_fstop
    return info


def place_pieces(
    rng: random.Random, args: argparse.Namespace, scene: bpy.types.Scene, templates: dict, position
) -> list[tuple[str, int, int, bpy.types.Object]]:
    s = args.square_size
    scale = rng.uniform(0.9, 1.1)
    placed = []
    for symbol, file, rank in position.pieces():
        dx, dy = (max(-0.25, min(0.25, rng.gauss(0, args.piece_jitter))) for _ in range(2))
        if symbol in "Nn" and rng.random() > 0.25:
            # Knights mostly face the opponent rather than spinning freely
            yaw = (0.0 if symbol == "N" else math.pi) + math.radians(rng.gauss(0, 25))
        else:
            yaw = rng.uniform(0, 2 * math.pi)
        obj = templates[symbol].copy()
        obj.hide_render = False
        obj.matrix_world = Matrix.LocRotScale(
            ((file - 3.5 + dx) * s, (rank - 3.5 + dy) * s, 0), Euler((0, 0, yaw)), (scale,) * 3
        )
        scene.collection.objects.link(obj)
        placed.append((symbol, file, rank, obj))
    return placed


def world_vertices(obj: bpy.types.Object, cache: dict) -> np.ndarray:
    mesh = obj.data
    if mesh.name not in cache:
        co = np.empty(len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", co)
        cache[mesh.name] = co.reshape(-1, 3)
    m = np.array(obj.matrix_world)
    return cache[mesh.name] @ m[:3, :3].T + m[:3, 3]


def find_sets(paths: list[Path]) -> list[Path]:
    sets = []
    for path in paths:
        if path.is_dir():
            sets += sorted(path.glob("*/scene.blend"))
            if (path / "scene.blend").is_file():
                sets.append(path / "scene.blend")
        elif path.is_file():
            sets.append(path)
        else:
            raise SystemExit(f"--sets path not found: {path}")
    if not sets:
        raise SystemExit(f"no scene.blend found in {[str(p) for p in paths]}")
    return sets


class Stage:
    """One opened chess set with the camera and light rig added to it."""

    def __init__(self, blend: Path | None, args: argparse.Namespace) -> None:
        if blend is None:
            bpy.ops.wm.read_factory_settings(use_empty=True)
        else:
            bpy.ops.wm.open_mainfile(filepath=str(blend))
        self.name = "procedural" if blend is None else blend.parent.name
        self.scene = scene = bpy.context.scene
        # Per-set copy of the arguments, since each set has its own square size
        self.args = argparse.Namespace(**vars(args))
        if args.square_size is None:
            # prepare_set.py stores the scale it normalised the set to
            self.args.square_size = float(scene.get("square_size", 0.05))
        self.device = setup_device(scene, args.device)
        setup_render(scene, args)
        square = self.args.square_size
        self.world = (ProceduralScene if blend is None else LoadedScene)(scene, square)
        self.lights = LightRig(scene)
        self.camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
        scene.collection.objects.link(self.camera)
        scene.camera = self.camera
        self.camera.data.clip_start = 0.01
        self.corners = np.array([[-4, -4, 0], [-4, 4, 0], [4, 4, 0], [4, -4, 0]], float) * square
        self.vertex_cache: dict = {}
        self.placed: list = []

    def render(self, index: int, position, image_path: Path) -> dict:
        args, scene, camera = self.args, self.scene, self.camera
        rng = random.Random(f"{args.seed}-{index}")
        white_turn = position.white_to_move
        if white_turn is None:
            white_turn = rng.random() < 0.5

        for *_, obj in self.placed:
            bpy.data.objects.remove(obj, do_unlink=True)
        looks = self.world.randomise(rng)
        self.placed = place_pieces(rng, args, scene, self.world.templates, position)
        piece_points = [world_vertices(obj, self.vertex_cache) for *_, obj in self.placed]
        must_see = np.concatenate([self.corners, *piece_points])
        camera_info = place_camera(rng, args, scene, camera, white_turn, must_see)
        lighting_info = self.lights.randomise(rng, args.square_size, camera.matrix_world)
        exposure = rng.gauss(0, 0.35)
        scene.view_settings.exposure = exposure
        scene.cycles.seed = index

        project = Projector(scene, camera)
        corner_px, _ = project(self.corners)
        pieces = []
        for (symbol, file, rank, _), points in zip(self.placed, piece_points):
            px, _ = project(points)
            (x0, y0), (x1, y1) = px.min(axis=0), px.max(axis=0)
            pieces.append(
                {
                    "piece": symbol,
                    "square": square_name(file, rank),
                    "box": [round(float(v), 1) for v in (x0, y0, x1 - x0, y1 - y0)],
                }
            )

        scene.render.filepath = str(image_path)
        bpy.ops.render.render(write_still=True)

        return {
            "fen": position.placement,
            "white_turn": white_turn,
            "camera": camera_info,
            "lighting": lighting_info,
            "exposure": exposure,
            "looks": looks,
            # a1, a8, h8, h1, as in ChessCog
            "corners": [[round(float(x), 1), round(float(y), 1)] for x, y in corner_px],
            "pieces": pieces,
            "render": {
                "seed": args.seed,
                "fen_line": index,
                "samples": args.samples,
                "resolution": [args.width, args.height],
                "scene": self.name,
                "square_size": args.square_size,
            },
        }


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = parse_args(argv)
    positions = read_positions(str(args.fens))
    end = len(positions) if args.end is None else min(args.end, len(positions))
    sets: list[Path | None] = [None] if args.procedural else list(find_sets(args.sets))

    # Pick a set per FEN from the seed and line number alone, then group by
    # set so each .blend is opened once
    jobs: dict[Path | None, list[tuple[int, Path, Path]]] = {}
    for index in range(args.start, end):
        split = assign_split(positions[index].placement, args.split_fractions)
        split_dir = args.out / "render" / split
        image_path, json_path = split_dir / f"{index:06d}.jpg", split_dir / f"{index:06d}.json"
        if json_path.exists():
            continue
        chosen = sets[random.Random(f"{args.seed}-{index}-set").randrange(len(sets))]
        jobs.setdefault(chosen, []).append((index, image_path, json_path))
    total = sum(len(j) for j in jobs.values())
    print(f"{total} FEN lines to render from {args.start}..{end - 1} over {len(jobs)} sets",
          flush=True)

    for blend, todo in jobs.items():
        stage = Stage(blend, args)
        print(f"set {stage.name}: {len(todo)} images on {stage.device}", flush=True)
        for index, image_path, json_path in todo:
            started = time.time()
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label = stage.render(index, positions[index], image_path)
            json_path.write_text(json.dumps(label, default=float))
            print(f"[{index}] {image_path.parent.name}/{image_path.name} {stage.name} "
                  f"{time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()

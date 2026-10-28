"""Pack the training datasets (ChessReD, ChessCog, ChessCam, SLCC, synthetic) into one tar with smaller images.

Images whose shorter side exceeds --max-side are shrunk to it (aspect kept) and every image is
stored as JPEG, so the tar holds no more pixels than training at that resolution uses: the
training transform resizes straight to a square, so a shorter side of 644 loses nothing at 644.
Annotations are copied unchanged and the layout matches data/, so the Modal training job
(chessqueries/train/training_job.py) extracts it and points DATA_ROOT at it.

Shrunk images are cached under --cache (one folder per --max-side and --quality) and reused while
their source is unchanged, so re-packing after adding a dataset only shrinks the new images.
The tar is written straight from the workers' output, with no unpacked copy on disk.

    .venv/bin/python scripts/pack_datasets.py            # -> dist/chessqueries-data.tar
"""

import argparse
import hashlib
import io
import json
import os
import tarfile
from multiprocessing import Pool
from pathlib import Path

from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
# Each dataset's folder under data/; SLCC's also holds its release bundle and source videos.
# The synthetic renders are the 5k set from synthetic/render.py (data/synthetic holds only its FENs)
DATASETS = {
    "chessred": "chessred",
    "chesscog": "chesscog",
    "chesscam": "chesscam",
    "slcc": "slcc/dataset",
    "synthetic": "synthetic_5k",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def shrink(job: tuple[Path, Path, int, int]) -> bytes:
    """The JPEG bytes for one image, from the cache when it was made from this exact source"""
    src, cached, max_side, quality = job
    src_mtime = src.stat().st_mtime_ns
    try:
        if cached.stat().st_mtime_ns == src_mtime:
            return cached.read_bytes()
    except FileNotFoundError:
        pass
    with Image.open(src) as im:
        if min(im.size) <= max_side and src.suffix.lower() in {".jpg", ".jpeg"}:
            # Already small enough and already JPEG: keep the original bytes rather than re-encode
            data = src.read_bytes()
        else:
            im = im.convert("RGB")
            short = min(im.size)
            if short > max_side:
                scale = max_side / short
                im = im.resize(
                    (round(im.width * scale), round(im.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality)
            data = buf.getvalue()
    # Write then rename so an interrupted run never leaves a truncated cache entry; the source's
    # mtime on the entry is what marks it as current
    cached.parent.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_name(f"{cached.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.utime(tmp, ns=(src_mtime, src_mtime))
    tmp.replace(cached)
    return data


def plan(src_root: Path, cache_root: Path, max_side: int, quality: int) -> tuple[list, list]:
    """(image jobs, other files to copy) for every dataset, each as (source, path under data/)"""
    jobs, copies = [], []
    for name, folder in DATASETS.items():
        root = src_root / folder
        if not root.is_dir():
            raise SystemExit(f"{root} not found; download or build {name} first")
        # Sorted so the tar's member order, and with it its sha256, is reproducible
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(src_root)
            if path.suffix.lower() in IMAGE_SUFFIXES:
                rel = rel.with_suffix(".jpg")
                jobs.append(((path, cache_root / rel, max_side, quality), rel))
            elif path.suffix == ".json":
                copies.append((path, rel))
    return jobs, copies


class HashingWriter:
    """File wrapper that hashes everything written through it, so the tar needs no second read"""

    def __init__(self, f):
        self.f = f
        self.digest = hashlib.sha256()

    def write(self, data) -> int:
        self.digest.update(data)
        return self.f.write(data)

    def tell(self) -> int:
        return self.f.tell()


def add_bytes(tar: tarfile.TarFile, name: str, data: bytes, mtime: float) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = int(mtime)
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "dist" / "chessqueries-data",
        help="Writes <out>.tar, whose top-level folder is named after <out>",
    )
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "dist" / ".pack-cache")
    parser.add_argument(
        "--max-side", type=int, default=644, help="Largest shorter side kept, in pixels"
    )
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality")
    args = parser.parse_args()

    cache_root = args.cache / f"{args.max_side}-q{args.quality}"
    jobs, copies = plan(args.src, cache_root, args.max_side, args.quality)
    top = args.out.name
    tar_path = args.out.parent / f"{top}.tar"
    partial = tar_path.with_name(f"{tar_path.name}.partial")
    tar_path.parent.mkdir(parents=True, exist_ok=True)

    # Uncompressed: the images are already JPEG, so gzip would only cost time
    with open(partial, "wb") as f:
        out = HashingWriter(f)
        with tarfile.open(fileobj=out, mode="w") as tar:
            for src, rel in copies:
                tar.add(src, arcname=f"{top}/{rel}")
            # Each image carries its source's mtime, so an unchanged data/ gives an identical tar
            newest = 0.0
            with Pool() as pool:
                results = pool.imap(shrink, [job for job, _ in jobs], chunksize=64)
                for (job, rel), data in tqdm(zip(jobs, results), total=len(jobs), desc="images"):
                    mtime = job[0].stat().st_mtime
                    newest = max(newest, mtime)
                    add_bytes(tar, f"{top}/{rel}", data, mtime)
            packing = {
                "datasets": DATASETS,
                "max_side": args.max_side,
                "jpeg_quality": args.quality,
                "images": len(jobs),
            }
            add_bytes(tar, f"{top}/PACKING.json", json.dumps(packing, indent=2).encode(), newest)
    partial.replace(tar_path)
    digest = out.digest.hexdigest()
    (tar_path.parent / f"{tar_path.name}.sha256").write_text(f"{digest}  {tar_path.name}\n")
    print(
        f"{len(jobs)} images -> {tar_path} ({tar_path.stat().st_size / 2**30:.2f} GiB), sha256 {digest}"
    )


if __name__ == "__main__":
    main()

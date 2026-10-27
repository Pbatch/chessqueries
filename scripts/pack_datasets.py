"""Pack the training datasets (ChessReD, ChessCog, ChessCam, SLCC) into one tar with smaller images.

Images whose shorter side exceeds --max-side are shrunk to it (aspect kept) and every image is
stored as JPEG, so the tar holds no more pixels than training at that resolution uses: the
training transform resizes straight to a square, so a shorter side of 644 loses nothing at 644.
Annotations are copied unchanged and the layout matches data/, so the Modal training job
(chessqueries/train/training_job.py) extracts it and points DATA_ROOT at it.

    .venv/bin/python scripts/pack_datasets.py            # -> dist/chessqueries-data.tar
"""
import argparse
import hashlib
import json
import shutil
import tarfile
from multiprocessing import Pool
from pathlib import Path

from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
# Each dataset's folder under data/; SLCC's also holds its release bundle and source videos
DATASETS = {"chessred": "chessred", "chesscog": "chesscog", "chesscam": "chesscam", "slcc": "slcc/dataset"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def shrink(job: tuple[Path, Path, int, int]) -> None:
    src, dest, max_side, quality = job
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        short = min(im.size)
        if short > max_side:
            scale = max_side / short
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        elif src.suffix.lower() in {".jpg", ".jpeg"}:
            # Already small enough and already JPEG: copy rather than re-encode
            shutil.copy2(src, dest)
            return
        im.save(dest, "JPEG", quality=quality)


def plan(src_root: Path, out_root: Path, max_side: int, quality: int) -> tuple[list, list]:
    """(image jobs, other files to copy) for every dataset"""
    jobs, copies = [], []
    for name, folder in DATASETS.items():
        root = src_root / folder
        if not root.is_dir():
            raise SystemExit(f"{root} not found; download or build {name} first")
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(src_root)
            if path.suffix.lower() in IMAGE_SUFFIXES:
                jobs.append((path, (out_root / rel).with_suffix(".jpg"), max_side, quality))
            elif path.suffix == ".json":
                copies.append((path, out_root / rel))
    return jobs, copies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "dist" / "chessqueries-data")
    parser.add_argument("--max-side", type=int, default=644, help="Largest shorter side kept, in pixels")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality")
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    jobs, copies = plan(args.src, args.out, args.max_side, args.quality)
    for src, dest in copies:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(shrink, jobs, chunksize=64), total=len(jobs), desc="images"):
            pass
    (args.out / "PACKING.json").write_text(json.dumps(
        {"datasets": DATASETS, "max_side": args.max_side, "jpeg_quality": args.quality,
         "images": len(jobs)}, indent=2))

    tar_path = args.out.parent / f"{args.out.name}.tar"
    # Uncompressed: the images are already JPEG, so gzip would only cost time
    with tarfile.open(tar_path, "w") as tar:
        tar.add(args.out, arcname=args.out.name)
    digest = hashlib.sha256()
    with open(tar_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    (tar_path.parent / f"{tar_path.name}.sha256").write_text(f"{digest.hexdigest()}  {tar_path.name}\n")
    print(f"{len(jobs)} images -> {tar_path} ({tar_path.stat().st_size / 2**30:.2f} GiB), sha256 {digest.hexdigest()}")


if __name__ == "__main__":
    main()

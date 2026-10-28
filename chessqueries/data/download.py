"""Dataset downloaders.

Each function is idempotent. File downloads use a ``.part`` path and become
visible at their final path only after any available SHA-256 check succeeds.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from types import SimpleNamespace

import requests
from tqdm import tqdm

from chessqueries.config import get_config
from chessqueries.data.base import DatasetName
from chessqueries.data.inventory import FROZEN_SAMPLE_COUNTS

# --- ChessReD (4TU.ResearchData) ------------------------------------------- #
CHESSRED_ANNOTATIONS_URL = (
    "https://data.4tu.nl/file/99b5c721-280b-450b-b058-b2900b69a90f/"
    "3cae6364-daca-4967-b426-1e4b68cdb64c"
)
CHESSRED_ANNOTATIONS_SHA256 = "16e99d7e8535c0fc56507caa2aa5f7594d7ca076ebab3f5a432cfd4aa10668cc"
# We use the authors' offline-resized 1024x1024 images (Google Drive zip), not
# the raw 3072x3072 4TU images: these are the exact inputs the ChessReD baseline
# was trained on, the only ones that reproduce the paper, and what we train our
# own models on too. See upstream issue tmasouris/end-to-end-chess-recognition#5.
CHESSRED_IMAGES_GDRIVE_ID = "1jxmFxjOy0qefdCZ_x3DMNtsvAK4LojEw"
CHESSRED_IMAGE_COUNT = sum(FROZEN_SAMPLE_COUNTS[DatasetName.CHESSRED].values())

# --- ChessCog (OSF) --------------------------------------------------------- #
# Synthetic render dataset, OSF project "xf3ka". Stored as val.zip, test.zip and
# a SPLIT archive train.zip + train.z01 (must be merged before unzip). Mirrors
# chesscog's own chesscog.data_synthesis.download_dataset.
CHESSCOG_OSF_PROJECT = "xf3ka"
CHESSCOG_SPLIT_COUNTS = {
    split.value: count
    for split, count in FROZEN_SAMPLE_COUNTS[DatasetName.CHESSCOG].items()
    if split is not None
}

# --- CVChess (Google Drive folder) ----------------------------------------- #
CVCHESS_GDRIVE_FOLDER_ID = "1lxS3GpYvDLU_1a9oqzIRlNmJUSYbmGZp"


class _ArchiveValidationError(RuntimeError):
    """A downloaded archive is readable enough to inspect but is invalid."""


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _part_path(dest: Path) -> Path:
    return dest.with_name(f"{dest.name}.part")


def _ready(dest: Path, expected_sha256: str | None) -> bool:
    if not dest.is_file() or dest.stat().st_size == 0:
        return False
    return expected_sha256 is None or _sha256(dest) == expected_sha256


def _prepare_part(dest: Path, expected_sha256: str | None) -> tuple[Path, bool]:
    """Return the partial path and whether a prior partial may be resumed."""
    part = _part_path(dest)
    if dest.exists() and not _ready(dest, expected_sha256):
        part.unlink(missing_ok=True)
        os.replace(dest, part)
        print(f"  [redo] {dest.name} failed validation")
        return part, False
    return part, part.is_file()


def _finish_part(
    part: Path,
    dest: Path,
    expected_sha256: str | None,
    *,
    discard_hash_mismatch: bool = False,
) -> Path:
    if not part.is_file() or part.stat().st_size == 0:
        raise RuntimeError(f"download produced no file at {part}")
    if expected_sha256:
        actual = _sha256(part)
        if actual != expected_sha256:
            if discard_hash_mismatch:
                part.unlink()
            raise RuntimeError(
                f"SHA-256 mismatch for {part} (expected {expected_sha256}, got {actual})"
            )
    os.replace(part, dest)
    return dest


def _download(url: str, dest: Path, expected_sha256: str | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _ready(dest, expected_sha256):
        print(f"  [skip] {dest.name} already present")
        return dest
    part, _ = _prepare_part(dest, expected_sha256)
    print(f"  [get ] {url}\n     -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with (
            open(part, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    return _finish_part(part, dest, expected_sha256)


def _download_gdrive(file_id: str, dest: Path, expected_sha256: str | None = None) -> Path:
    import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)
    if _ready(dest, expected_sha256):
        print(f"  [skip] {dest.name} already present")
        return dest
    part, resume = _prepare_part(dest, expected_sha256)
    if not resume:
        part.unlink(missing_ok=True)
    result = gdown.download(
        id=file_id,
        output=str(part),
        quiet=False,
        resume=resume,
    )
    if result is None:
        raise RuntimeError(f"Google Drive download failed for {dest.name}")
    return _finish_part(part, dest, expected_sha256, discard_hash_mismatch=True)


def _extract_zip(zip_path: Path, out_dir: Path, expected_root: str) -> None:
    """Extract one dataset directory and replace it only after a complete unzip."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{expected_root}.part-", dir=out_dir))
    target = out_dir / expected_root
    backup: Path | None = None
    print(f"  [unzip] {zip_path.name} -> {target}")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            corrupt = zf.testzip()
            if corrupt is not None:
                raise _ArchiveValidationError(f"corrupt member in {zip_path}: {corrupt}")
            for name in tqdm(zf.namelist(), desc="extract", unit="file"):
                zf.extract(name, stage)
        extracted = stage / expected_root
        if not extracted.is_dir():
            raise _ArchiveValidationError(
                f"{zip_path} did not contain the expected {expected_root}/ directory"
            )
        if target.exists():
            backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
            os.replace(target, backup)
        try:
            os.replace(extracted, target)
        except Exception:
            if backup is not None and not target.exists():
                os.replace(backup, target)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _paired_file_count(directory: Path, annotation_suffix: str, *image_suffixes: str) -> int:
    annotations = {path.stem for path in directory.glob(f"*{annotation_suffix}")}
    images = {path.stem for suffix in image_suffixes for path in directory.glob(f"*{suffix}")}
    return len(annotations) if annotations == images else -1


# The renders ship as PNG; data/chesscog's have been converted to quality-95 JPEG in place
CHESSCOG_IMAGE_SUFFIXES = (".png", ".jpg")


def _chesscog_complete(render: Path) -> bool:
    return all(
        _paired_file_count(render / split, ".json", *CHESSCOG_IMAGE_SUFFIXES) == expected
        for split, expected in CHESSCOG_SPLIT_COUNTS.items()
    )


def download_chessred(dataroot: Path | None = None) -> Path:
    """Download ChessReD: annotations (4TU) + the authors' 1024x1024 images.

    The images are the offline-resized set the released checkpoint was trained on
    (see CHESSRED_IMAGES_GDRIVE_ID); they extract to ``<dataroot>/images/``.
    """
    dataroot = Path(dataroot) if dataroot else get_config().chessred_root
    print("ChessReD ->", dataroot)
    _download(
        CHESSRED_ANNOTATIONS_URL,
        dataroot / "annotations.json",
        CHESSRED_ANNOTATIONS_SHA256,
    )
    images_dir = dataroot / "images"
    if images_dir.is_dir() and len(list(images_dir.rglob("*.jpg"))) == CHESSRED_IMAGE_COUNT:
        print("  [skip] images/ already extracted")
    else:
        dataroot.mkdir(parents=True, exist_ok=True)
        zip_path = dataroot / "images_1024.zip"
        _download_gdrive(CHESSRED_IMAGES_GDRIVE_ID, zip_path)
        try:
            _extract_zip(zip_path, dataroot, "images")
        except (zipfile.BadZipFile, _ArchiveValidationError) as exc:
            zip_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"ChessReD image archive {zip_path} failed ZIP validation and was "
                "removed; rerun the download to fetch a fresh copy"
            ) from exc
        zip_path.unlink(missing_ok=True)
        image_count = len(list(images_dir.rglob("*.jpg")))
        if image_count != CHESSRED_IMAGE_COUNT:
            raise RuntimeError(
                f"ChessReD extraction produced {image_count} images, "
                f"expected {CHESSRED_IMAGE_COUNT}"
            )
    return dataroot


def download_chesscog(dataroot: Path | None = None) -> Path:
    """Clone the OSF project and unpack val/test/train (train is split-zip)."""
    import subprocess

    from osfclient import cli

    dataroot = Path(dataroot) if dataroot else get_config().chesscog_root
    render = dataroot / "render"
    if _chesscog_complete(render):
        print(f"  [skip] complete ChessCog render already present at {render}")
        return dataroot
    print("ChessCog (synthetic) ->", render)

    # 1. Clone OSF storage away from the final tree, then install its archives.
    render.mkdir(parents=True, exist_ok=True)
    clone_stage = Path(tempfile.mkdtemp(prefix=".chesscog-download.part-", dir=dataroot.parent))
    try:
        args = SimpleNamespace(
            project=CHESSCOG_OSF_PROJECT,
            output=str(clone_stage),
            username=None,
        )
        cli.clone(args)
        osfstorage = clone_stage / "osfstorage"
        if not osfstorage.is_dir():
            raise RuntimeError("OSF clone produced no osfstorage directory")
        for item in osfstorage.iterdir():
            os.replace(item, render / item.name)
    finally:
        shutil.rmtree(clone_stage, ignore_errors=True)

    # 2. Unzip val/test directly.
    for archive in ("val.zip", "test.zip"):
        if (render / archive).exists():
            _extract_zip(render / archive, render, Path(archive).stem)

    # 3. Merge the split train archive (train.zip + train.z01) then unzip.
    if (render / "train.zip").exists() and not (render / "train").is_dir():
        merged = render / "train_full.zip"
        rc = subprocess.run(
            ["zip", "-s", "0", str(render / "train.zip"), "--out", str(merged)]
        ).returncode
        if rc != 0 or not merged.exists():
            raise RuntimeError(
                f"Could not merge split train archive in {render}. Merge train.zip "
                f"+ train.z01 manually (e.g. `zip -s 0 train.zip --out train_full.zip`) "
                f"and unzip."
            )
        _extract_zip(merged, render, "train")
        for f in ("train.z01", "train.zip", "train_full.zip", "val.zip", "test.zip"):
            (render / f).unlink(missing_ok=True)
    if not _chesscog_complete(render):
        counts = {
            split: _paired_file_count(render / split, ".json", *CHESSCOG_IMAGE_SUFFIXES)
            for split in CHESSCOG_SPLIT_COUNTS
        }
        raise RuntimeError(
            f"ChessCog extraction is incomplete: paired-file counts {counts}, "
            f"expected {CHESSCOG_SPLIT_COUNTS}"
        )
    return dataroot


def download_cvchess(dataroot: Path | None = None) -> Path:
    """Seed CVChess into the standard layout: ``data/cvchess/{images/, annotations.json}``.

    The CVChess Drive folder has >50 files, which exceeds gdown's web-scraping
    limit (listing needs the authenticated Drive API), so the **images must be
    downloaded manually**. This function seeds ``annotations.json`` from the
    vendored labels and tells you where to drop the images.
    """
    from chessqueries.data.cvchess import VENDORED_LABELS

    dataroot = Path(dataroot) if dataroot else get_config().cvchess_root
    images_dir = dataroot / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Seed annotations.json (alongside images, like the other datasets).
    ann = dataroot / "annotations.json"
    if not ann.exists():
        shutil.copy(VENDORED_LABELS, ann)
        print(f"  [seed] annotations.json -> {ann}")

    n_images = len(list(images_dir.glob("*.jpg")))
    if n_images == 0:
        print(
            "CVChess images must be downloaded manually (Drive folder has >50 files,\n"
            "which defeats gdown). Download from\n"
            f"  https://drive.google.com/drive/folders/{CVCHESS_GDRIVE_FOLDER_ID}\n"
            f"and place the IMG_*.jpg files in:\n  {images_dir}\n"
            "Annotations are already in place."
        )
    else:
        print(f"  [ok] {n_images} images present in {images_dir}")
    return dataroot


# Downloadable datasets (SLCC is reconstructed with the upstream tooling, ChessCam built elsewhere).
DATASETS = {
    DatasetName.CHESSRED: download_chessred,
    DatasetName.CHESSCOG: download_chesscog,
    DatasetName.CVCHESS: download_cvchess,
}

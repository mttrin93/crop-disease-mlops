"""
Preprocess raw PlantVillage images for EfficientNet-B0 training.

  1. Scans data/raw/color/ for class directories
  2. Creates leaf-aware train / val / test split (70 / 15 / 15)
  3. Resizes all images to 224x224 (EfficientNet-B0 input size)
  4. Saves processed images + metadata.json to data/processed/

Leaf-aware splitting:
  PlantVillage filenames encode a leaf_id UUID:
    a37f34f7-022d-461a-8a3d-95f5cd774e35___Mary_HL 9155.JPG
    └─── leaf_id ────────────────────────┘

  The same physical leaf can appear multiple times (different capture angles,
  lighting conditions). Splitting on individual images risks putting photos of
  the same leaf in both train and test — the model would have "seen" the leaf
  during training, artificially inflating test accuracy.

  We split on leaf_ids instead: all images of the same leaf end up in the
  same split. If filenames don't follow the convention, each image is treated
  as its own leaf (safe fallback, no leakage possible).

No S3 here — uploading is handled by the Airflow DAG.

Usage:
    uv run python src/data/preprocess.py
"""

import os
import json
import random
import logging
import argparse
from pathlib import Path

import boto3
from PIL import Image
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
)
logger = logging.getLogger(__name__)

LOCAL_RAW_DIR = Path("data/raw/color")
LOCAL_PROCESSED_DIR = Path("data/processed")

IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

S3_PROCESSED_PREFIX = "data/processed"

# ---------- data loading ----------


def scan_dataset(raw_dir: Path) -> tuple[list[str], dict[str, list[Path]]]:
    """
    Scan raw_dir for class subdirectories and collect all image paths.

    Returns:
        class_names: sorted list of class name strings
        image_map:   dict mapping class_name → list of image Paths
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}\n"
            "Unzip the PlantVillage archive under data/raw/color/"
        )

    image_map: dict[str, list[Path]] = {}
    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        paths = sorted(
            p
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix in IMAGE_EXTENSIONS
        )
        if paths:
            image_map[class_dir.name] = paths
        else:
            logger.warning("No images in %s, skipping", class_dir.name)

    if not image_map:
        raise ValueError(
            f"No valid class directories found in {raw_dir}. "
            "Check that the directory contains subdirectories with images."
        )

    class_names = sorted(image_map.keys())
    total = sum(len(v) for v in image_map.values())
    logger.info("Found %s classes, %s total images", len(class_names), total)
    return class_names, image_map


# ---------- leaf-aware splitting ----------


def extract_leaf_id(image_path: Path) -> str:
    """
    Extract leaf_id from a PlantVillage filename.

    PlantVillage convention:
        <leaf_id>___<capture_info>.<ext>
        e.g. a37f34f7-022d-461a-8a3d-95f5cd774e35___Mary_HL 9155.JPG
              └─── leaf_id ────────────────────────┘

    Returns the leaf_id string if found, otherwise the full stem
    (treating the image as its own unique leaf — safe fallback).
    """
    stem = image_path.stem
    if "___" in stem:
        return stem.split("___")[0]
    return stem  # fallback: treat each image as a distinct leaf


def group_by_leaf(paths: list[Path]) -> dict[str, list[Path]]:
    """Group image paths by their leaf_id."""
    leaf_map: dict[str, list[Path]] = {}
    for path in paths:
        leaf_id = extract_leaf_id(path)
        leaf_map.setdefault(leaf_id, []).append(path)
    return leaf_map


class _SplitAccumulator:  # pylint: disable=too-few-public-methods
    """Groups mutable split state to avoid long argument lists."""

    def __init__(self) -> None:
        self.splits: dict[str, list[tuple[Path, str]]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        self.leaf_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    def add(
        self,
        leaf_map: dict[str, list[Path]],
        leaves: list[str],
        split_name: str,
        class_name: str,
    ) -> None:
        """Add all images for the given leaves into the named split."""
        for leaf_id in leaves:
            self.splits[split_name].extend([(p, class_name) for p in leaf_map[leaf_id]])
            self.leaf_counts[split_name] += 1


def _compute_leaf_splits(
    unique_leaves: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Split leaf IDs into train / val / test lists."""
    train_leaves, remainder = train_test_split(
        unique_leaves,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
    )
    val_ratio_adjusted = val_ratio / (val_ratio + TEST_RATIO)
    val_leaves, test_leaves = train_test_split(
        remainder,
        train_size=val_ratio_adjusted,
        random_state=seed,
        shuffle=True,
    )
    return train_leaves, val_leaves, test_leaves


def create_splits(
    image_map: dict[str, list[Path]],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> dict[str, list[tuple[Path, str]]]:
    """
    Create leaf-aware stratified train / val / test splits.

    Splits are done on leaf_ids (not individual images), so all photos of
    the same physical leaf always end up in the same split. This prevents
    data leakage where the model "sees" a leaf during training and then
    evaluates on another photo of the same leaf.

    Returns dict with keys 'train', 'val', 'test', each a list of
    (image_path, class_name) tuples.
    """
    acc = _SplitAccumulator()

    for class_name, paths in image_map.items():
        leaf_map = group_by_leaf(paths)
        unique_leaves = sorted(leaf_map.keys())
        train_leaves, val_leaves, test_leaves = _compute_leaf_splits(
            unique_leaves, train_ratio, val_ratio, seed
        )
        acc.add(leaf_map, train_leaves, "train", class_name)
        acc.add(leaf_map, val_leaves, "val", class_name)
        acc.add(leaf_map, test_leaves, "test", class_name)

    # shuffle images within each split
    for _, split_items in acc.splits.items():
        random.seed(seed)
        random.shuffle(split_items)

    logger.info(
        "Leaf-aware splits: "
        "train=%s leaves / %s images, "
        "val=%s leaves / %s images, "
        "test=%s leaves / %s images",
        acc.leaf_counts["train"],
        len(acc.splits["train"]),
        acc.leaf_counts["val"],
        len(acc.splits["val"]),
        acc.leaf_counts["test"],
        len(acc.splits["test"]),
    )
    return acc.splits


# ---------- image processing ----------


def resize_image(image: Image.Image, size: tuple[int, int] = IMAGE_SIZE) -> Image.Image:
    """
    Resize to target size with Lanczos resampling.

    PIL is used here (not torchvision) because preprocess.py runs without
    PyTorch. Augmentation transforms are applied in the training notebook
    via torchvision.transforms.
    """
    return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def save_splits(
    splits: dict[str, list[tuple[Path, str]]],
    output_dir: Path,
) -> dict[str, dict[str, int]]:
    """
    Resize and save images as output_dir/<split>/<class_name>/<index>.jpg.
    Returns per-split, per-class image counts.
    """
    counts: dict[str, dict[str, int]] = {split: {} for split in splits}

    for split_name, items in splits.items():
        logger.info("[%s] Saving %s images...", split_name, len(items))

        for i, (src_path, class_name) in enumerate(items):
            dest_dir = output_dir / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            dest_path = dest_dir / f"{i:06d}.jpg"
            resize_image(Image.open(src_path)).save(dest_path, "JPEG", quality=95)

            counts[split_name][class_name] = counts[split_name].get(class_name, 0) + 1

            if (i + 1) % 1000 == 0:
                logger.info("  [%s] %s/%s saved", split_name, i + 1, len(items))

    return counts


# ---------- metadata ----------


def save_metadata(
    output_dir: Path,
    class_names: list[str],
    counts: dict[str, dict[str, int]],
) -> None:
    """
    Save metadata.json alongside the processed splits.
    Read by the Colab training notebook and the FastAPI inference service.
    """
    metadata = {
        "image_size": list(IMAGE_SIZE),
        "class_names": class_names,
        "class_to_index": {name: i for i, name in enumerate(class_names)},
        "index_to_class": {str(i): name for i, name in enumerate(class_names)},
        "num_classes": len(class_names),
        "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "random_seed": RANDOM_SEED,
        "leaf_aware_split": True,
        "splits": {
            split: {"total": sum(c.values()), "counts_per_class": c}
            for split, c in counts.items()
        },
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Metadata saved → %s", metadata_path)
    for split, info in metadata["splits"].items():
        logger.info("  %s: %s images", split, info["total"])


# ---------- upload to S3 for colab ----------


def upload_to_s3(local_dir: Path, bucket: str, prefix: str) -> None:
    """Upload processed splits to S3."""
    s3_client = boto3.client("s3")
    files = [f for f in local_dir.rglob("*") if f.is_file()]
    logger.info(
        "Uploading %s processed files to s3://%s/%s/", len(files), bucket, prefix
    )

    for i, file_path in enumerate(files):
        s3_key = f"{prefix}/{file_path.relative_to(local_dir)}"
        s3_client.upload_file(str(file_path), bucket, s3_key)

        if (i + 1) % 500 == 0:
            logger.info("Uploaded %s/%s files", i + 1, len(files))

    logger.info("Upload complete → s3://%s/%s/", bucket, prefix)


# ---------- entry point ----------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess PlantVillage dataset")
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip S3 upload (local preprocessing only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_bucket = os.getenv("MODEL_BUCKET")

    class_names, image_map = scan_dataset(LOCAL_RAW_DIR)

    splits = create_splits(image_map)

    LOCAL_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    counts = save_splits(splits, LOCAL_PROCESSED_DIR)
    save_metadata(LOCAL_PROCESSED_DIR, class_names, counts)

    logger.info("Done → %s/", LOCAL_PROCESSED_DIR)

    # upload to S3 bucket
    if not args.skip_upload:
        upload_to_s3(LOCAL_PROCESSED_DIR, model_bucket, S3_PROCESSED_PREFIX)
    else:
        logger.info("Skipping S3 upload (--skip-upload flag set)")


if __name__ == "__main__":
    main()

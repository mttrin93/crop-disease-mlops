"""Unit tests for src/data/preprocess.py."""

# pylint: disable=redefined-outer-name

import json
from pathlib import Path

import pytest
from PIL import Image

from src.data.preprocess import (
    VAL_RATIO,
    IMAGE_SIZE,
    TEST_RATIO,
    RANDOM_SEED,
    TRAIN_RATIO,
    save_splits,
    resize_image,
    scan_dataset,
    create_splits,
    group_by_leaf,
    save_metadata,
    extract_leaf_id,
)

# ---------- fixtures ----------


@pytest.fixture()
def sample_raw_dir(tmp_path: Path) -> Path:
    """
    Minimal PlantVillage-style directory with 3 classes.
    Filenames follow PlantVillage convention: <leaf_id>___<info>.JPG
    Each class has 5 leaves × 2 images = 10 images.
    """
    classes = ["Tomato___healthy", "Tomato___Late_blight", "Apple___healthy"]
    for class_name in classes:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for leaf_idx in range(5):
            leaf_id = f"leaf{leaf_idx:04d}-0000-0000-0000-000000000000"
            for shot in range(2):
                img = Image.new(
                    "RGB", (256, 256), color=(leaf_idx * 40, shot * 80, 100)
                )
                img.save(class_dir / f"{leaf_id}___RS_shot{shot}.JPG", "JPEG")
    return tmp_path


@pytest.fixture()
def sample_image_map(sample_raw_dir: Path) -> dict:
    _, image_map = scan_dataset(sample_raw_dir)
    return image_map


# ---------- extract_leaf_id ----------


def test_extract_leaf_id_standard_filename():
    path = Path("a37f34f7-022d-461a-8a3d-95f5cd774e35___Mary_HL 9155.JPG")
    assert extract_leaf_id(path) == "a37f34f7-022d-461a-8a3d-95f5cd774e35"


def test_extract_leaf_id_fallback_no_separator():
    path = Path("image_001.jpg")
    assert extract_leaf_id(path) == "image_001"


def test_extract_leaf_id_multiple_separators():
    path = Path("abc123___info___extra.JPG")
    assert extract_leaf_id(path) == "abc123"


def test_extract_leaf_id_returns_string():
    path = Path("someleaf___capture.jpg")
    assert isinstance(extract_leaf_id(path), str)


# ---------- group_by_leaf ----------


def test_group_by_leaf_groups_correctly(sample_raw_dir: Path):
    class_dir = sample_raw_dir / "Tomato___healthy"
    paths = list(class_dir.iterdir())
    leaf_map = group_by_leaf(paths)
    # 5 leaves × 2 images = 10 images, 5 unique leaf_ids
    assert len(leaf_map) == 5
    for leaf_paths in leaf_map.values():
        assert len(leaf_paths) == 2


def test_group_by_leaf_single_image(tmp_path: Path):
    img_path = tmp_path / "abc___info.jpg"
    img_path.write_bytes(b"fake")
    leaf_map = group_by_leaf([img_path])
    assert list(leaf_map.keys()) == ["abc"]


# ---------- scan_dataset ----------


def test_scan_dataset_finds_all_classes(sample_raw_dir):
    class_names, _ = scan_dataset(sample_raw_dir)
    assert set(class_names) == {
        "Tomato___healthy",
        "Tomato___Late_blight",
        "Apple___healthy",
    }


def test_scan_dataset_sorted(sample_raw_dir):
    class_names, _ = scan_dataset(sample_raw_dir)
    assert class_names == sorted(class_names)


def test_scan_dataset_correct_image_count(sample_raw_dir):
    _, image_map = scan_dataset(sample_raw_dir)
    for paths in image_map.values():
        assert len(paths) == 10  # 5 leaves × 2 images


def test_scan_dataset_handles_uppercase_extension(tmp_path):
    (tmp_path / "ClassA").mkdir()
    Image.new("RGB", (64, 64)).save(tmp_path / "ClassA" / "leaf1___info.JPG")
    Image.new("RGB", (64, 64)).save(tmp_path / "ClassA" / "leaf2___info.jpg")
    _, image_map = scan_dataset(tmp_path)
    assert len(image_map["ClassA"]) == 2


def test_scan_dataset_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        scan_dataset(Path("nonexistent/path"))


def test_scan_dataset_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError):
        scan_dataset(tmp_path)


# ---------- create_splits — leaf-awareness ----------


def test_create_splits_no_leaf_appears_in_multiple_splits(sample_image_map):
    """Core guarantee: the same leaf_id must never appear in two different splits."""
    splits = create_splits(sample_image_map)
    leaf_ids_per_split = {
        split_name: {extract_leaf_id(p) for p, _ in items}
        for split_name, items in splits.items()
    }
    assert leaf_ids_per_split["train"].isdisjoint(leaf_ids_per_split["val"])
    assert leaf_ids_per_split["train"].isdisjoint(leaf_ids_per_split["test"])
    assert leaf_ids_per_split["val"].isdisjoint(leaf_ids_per_split["test"])


def test_create_splits_returns_three_keys(sample_image_map):
    splits = create_splits(sample_image_map)
    assert set(splits.keys()) == {"train", "val", "test"}


def test_create_splits_covers_all_images(sample_image_map):
    splits = create_splits(sample_image_map)
    total_original = sum(len(v) for v in sample_image_map.values())
    total_split = sum(len(v) for v in splits.values())
    assert total_split == total_original


def test_create_splits_approximate_ratios(sample_image_map):
    splits = create_splits(sample_image_map)
    total = sum(len(v) for v in splits.values())
    assert abs(len(splits["train"]) / total - TRAIN_RATIO) < 0.10
    assert abs(len(splits["val"]) / total - VAL_RATIO) < 0.10
    assert abs(len(splits["test"]) / total - TEST_RATIO) < 0.10


def test_create_splits_reproducible(sample_image_map):
    a = create_splits(sample_image_map, seed=RANDOM_SEED)
    b = create_splits(sample_image_map, seed=RANDOM_SEED)
    assert [str(p) for p, _ in a["train"]] == [str(p) for p, _ in b["train"]]


def test_create_splits_all_classes_in_all_splits(sample_image_map):
    splits = create_splits(sample_image_map)
    for split_name, items in splits.items():
        classes_present = {c for _, c in items}
        for class_name in sample_image_map:
            assert (
                class_name in classes_present
            ), f"Class '{class_name}' missing from '{split_name}' split"


# ---------- resize_image ----------


def test_resize_image_output_size():
    assert resize_image(Image.new("RGB", (512, 384))).size == IMAGE_SIZE


def test_resize_image_converts_to_rgb():
    assert resize_image(Image.new("RGBA", (300, 300))).mode == "RGB"


def test_resize_image_small_input():
    assert resize_image(Image.new("RGB", (32, 32))).size == IMAGE_SIZE


# ---------- save_splits ----------


def test_save_splits_creates_files(sample_image_map, tmp_path):
    splits = create_splits(sample_image_map)
    save_splits(splits, tmp_path)
    saved = list(tmp_path.rglob("*.jpg"))
    assert len(saved) == sum(len(v) for v in sample_image_map.values())


def test_save_splits_correct_directory_structure(sample_image_map, tmp_path):
    splits = create_splits(sample_image_map)
    save_splits(splits, tmp_path)
    for split_name in ["train", "val", "test"]:
        assert (tmp_path / split_name).exists()


def test_save_splits_images_correct_size(sample_image_map, tmp_path):
    splits = create_splits(sample_image_map)
    save_splits({"train": splits["train"][:5]}, tmp_path)
    for img_path in (tmp_path / "train").rglob("*.jpg"):
        assert Image.open(img_path).size == IMAGE_SIZE


# ---------- save_metadata ----------


def test_save_metadata_creates_file(tmp_path):
    counts = {"train": {"A": 7}, "val": {"A": 2}, "test": {"A": 1}}
    save_metadata(tmp_path, ["A"], counts)
    assert (tmp_path / "metadata.json").exists()


def test_save_metadata_has_leaf_aware_flag(tmp_path):
    save_metadata(
        tmp_path, ["A"], {"train": {"A": 7}, "val": {"A": 2}, "test": {"A": 1}}
    )
    with open(tmp_path / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["leaf_aware_split"] is True


def test_save_metadata_index_to_class(tmp_path):
    class_names = ["Apple___healthy", "Tomato___healthy"]
    save_metadata(tmp_path, class_names, {"train": {}, "val": {}, "test": {}})
    with open(tmp_path / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["index_to_class"]["0"] == "Apple___healthy"
    assert meta["index_to_class"]["1"] == "Tomato___healthy"


def test_save_metadata_class_to_index(tmp_path):
    class_names = ["Apple___healthy", "Tomato___healthy"]
    save_metadata(tmp_path, class_names, {"train": {}, "val": {}, "test": {}})
    with open(tmp_path / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["class_to_index"]["Apple___healthy"] == 0
    assert meta["class_to_index"]["Tomato___healthy"] == 1

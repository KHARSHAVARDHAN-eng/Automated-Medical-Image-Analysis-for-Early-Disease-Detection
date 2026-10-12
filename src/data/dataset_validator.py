"""Dataset validation module for YOLO object detection.

Validates image-label pairing, bounding box normalization and validity,
empty label files for negative samples, and patient split integrity (zero data leakage).
"""

from pathlib import Path
from typing import Dict, List, Set, Any, Union, Optional
import os


def validate_splits_leakage(patient_split_map: Dict[str, str]) -> Dict[str, Any]:
    """Verify that patient IDs across train, val, and test partitions are strictly disjoint.

    Args:
        patient_split_map: Mapping of patientId -> split ('train', 'val', 'test').

    Returns:
        Dict[str, Any]:
            - "is_valid": bool
            - "overlap_count": int
            - "split_counts": Dict[str, int]
            - "errors": List[str]
    """
    splits: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}

    for pid, split in patient_split_map.items():
        if split in splits:
            splits[split].add(pid)
        else:
            return {
                "is_valid": False,
                "overlap_count": 0,
                "split_counts": {},
                "errors": [f"Unrecognized split name: '{split}' for patient {pid}"]
            }

    errors: List[str] = []
    # Check pair-wise intersections
    train_val_overlap = splits["train"].intersection(splits["val"])
    train_test_overlap = splits["train"].intersection(splits["test"])
    val_test_overlap = splits["val"].intersection(splits["test"])

    if train_val_overlap:
        errors.append(f"Patient leakage between Train and Val: {len(train_val_overlap)} shared patients.")
    if train_test_overlap:
        errors.append(f"Patient leakage between Train and Test: {len(train_test_overlap)} shared patients.")
    if val_test_overlap:
        errors.append(f"Patient leakage between Val and Test: {len(val_test_overlap)} shared patients.")

    return {
        "is_valid": len(errors) == 0,
        "overlap_count": len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "errors": errors,
    }


def validate_yolo_label_file(label_path: Path) -> Dict[str, Any]:
    """Validate a single YOLO .txt label file for bounding box correctness.

    Empty files are completely valid (representing negative background cases).
    Non-empty files must contain 5 normalized floats per row:
    <class_id> <x_center> <y_center> <width> <height>
    """
    errors: List[str] = []
    boxes: List[List[float]] = []

    content = label_path.read_text().strip()
    if not content:
        # Negative case (0 boxes) -> fully valid
        return {"is_valid": True, "box_count": 0, "boxes": [], "errors": []}

    lines = content.splitlines()
    for idx, line in enumerate(lines, start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            errors.append(f"Line {idx}: Expected 5 elements, found {len(parts)} ('{line}')")
            continue

        try:
            cls_id = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError as e:
            errors.append(f"Line {idx}: Non-numeric token in '{line}': {e}")
            continue

        if cls_id != 0:
            errors.append(f"Line {idx}: Unexpected class_id {cls_id} (expected 0 for pneumonia_opacity)")

        # Validate bounding box bounds: normalized coordinates in [0, 1]
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            errors.append(f"Line {idx}: Center ({xc}, {yc}) out of bounds [0, 1]")
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            errors.append(f"Line {idx}: Dimension width={w}, height={h} out of bounds (0, 1]")

        boxes.append([cls_id, xc, yc, w, h])

    return {
        "is_valid": len(errors) == 0,
        "box_count": len(boxes),
        "boxes": boxes,
        "errors": errors,
    }


def validate_yolo_dataset(
    dataset_dir: Union[str, Path],
    require_images: bool = False
) -> Dict[str, Any]:
    """Comprehensive integrity validation of a prepared YOLO dataset directory.

    Checks:
    1. Structure: existence of labels/ (and optionally images/).
    2. Label formats: empty for negatives, valid 5-token normalized boxes for positives.
    3. Pairing: 1-to-1 match between images and labels if require_images is True.
    4. Dataset statistics: counts per split, positive/negative distribution.

    Args:
        dataset_dir: Path to prepared YOLO dataset root.
        require_images: If True, checks that images/ exist and match labels 1:1.

    Returns:
        Dict[str, Any]: Validation summary, integrity status, and per-split statistics.
    """
    root = Path(dataset_dir)
    labels_root = root / "labels"

    stats: Dict[str, Any] = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "splits": {}
    }

    if not labels_root.exists():
        stats["is_valid"] = False
        stats["errors"].append(f"Missing labels directory: {labels_root}")
        return stats

    for split in ["train", "val", "test"]:
        split_labels_dir = labels_root / split
        if not split_labels_dir.exists():
            continue

        label_files = list(split_labels_dir.glob("*.txt"))
        total_samples = len(label_files)
        positive_samples = 0
        negative_samples = 0
        total_boxes = 0
        split_errors = []

        for lf in label_files:
            res = validate_yolo_label_file(lf)
            if not res["is_valid"]:
                split_errors.extend([f"{lf.name}: {err}" for err in res["errors"]])

            if res["box_count"] > 0:
                positive_samples += 1
                total_boxes += res["box_count"]
            else:
                negative_samples += 1

        # Check image pairing if requested
        if require_images:
            split_images_dir = root / "images" / split
            if not split_images_dir.exists():
                split_errors.append(f"Missing images directory for split: {split_images_dir}")
            else:
                img_stems = {img.stem for img in split_images_dir.glob("*.*") if img.suffix.lower() in [".png", ".jpg", ".jpeg"]}
                lbl_stems = {lbl.stem for lbl in label_files}

                missing_imgs = lbl_stems - img_stems
                missing_lbls = img_stems - lbl_stems

                if missing_imgs:
                    split_errors.append(f"{len(missing_imgs)} labels without matching images in {split}.")
                if missing_lbls:
                    split_errors.append(f"{len(missing_lbls)} images without matching labels in {split}.")

        stats["splits"][split] = {
            "total_samples": total_samples,
            "positive_samples": positive_samples,
            "negative_samples": negative_samples,
            "total_boxes": total_boxes,
            "error_count": len(split_errors),
        }

        if split_errors:
            stats["is_valid"] = False
            stats["errors"].extend(split_errors)

    return stats

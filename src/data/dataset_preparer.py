"""Reproducible dataset preparation pipeline for RSNA Pneumonia Detection.

Performs patient-level stratified train/val/test splitting to prevent patient data leakage,
converts RSNA bounding boxes to normalized YOLO format, generates negative (empty) label files,
and exports the dataset.yaml configuration.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import os
import yaml
import pandas as pd
import numpy as np

from src.config import CLASS_NAME_PNEUMONIA
from src.data.rsna_parser import convert_bbox_to_yolo


def split_patients(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42
) -> Dict[str, str]:
    """Perform patient-level stratified train/val/test split.

    Groups rows by patientId and stratifies based on whether the patient has
    at least one pneumonia opacity (Target=1) or no opacities (Target=0).
    Guarantees zero patient overlap/leakage across splits.

    Args:
        df: DataFrame containing at least 'patientId' and 'Target' columns.
        train_ratio: Fraction of patients allocated to train (default: 0.80).
        val_ratio: Fraction of patients allocated to validation (default: 0.10).
        test_ratio: Fraction of patients allocated to test (default: 0.10).
        random_seed: Random state for deterministic reproducibility.

    Returns:
        Dict[str, str]: Mapping from patientId to split name ('train', 'val', or 'test').
    """
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got: {total}")

    # Patient-level aggregation: max(Target) -> 1 if patient has pneumonia, else 0
    patient_targets = df.groupby("patientId")["Target"].max()

    # Stratified split by target
    rng = np.random.RandomState(random_seed)
    patient_split_map: Dict[str, str] = {}

    for target_val in [0, 1]:
        matching_patients = patient_targets[patient_targets == target_val].index.to_numpy()
        rng.shuffle(matching_patients)

        n_total = len(matching_patients)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))

        # Adjust for edge cases if small sample
        if n_total > 0 and n_train == 0:
            n_train = 1

        train_pts = matching_patients[:n_train]
        val_pts = matching_patients[n_train:n_train + n_val]
        test_pts = matching_patients[n_train + n_val:]

        for pid in train_pts:
            patient_split_map[pid] = "train"
        for pid in val_pts:
            patient_split_map[pid] = "val"
        for pid in test_pts:
            patient_split_map[pid] = "test"

    return patient_split_map


def generate_yolo_labels(
    df: pd.DataFrame,
    patient_split_map: Dict[str, str],
    output_dir: Union[str, Path],
    img_width: float = 1024.0,
    img_height: float = 1024.0,
    class_id: int = 0
) -> Dict[str, int]:
    """Generate YOLO format .txt label files for all patients.

    Positive cases receive bounding boxes:
      <class_id> <x_center_norm> <y_center_norm> <w_norm> <h_norm>
    Negative cases receive an empty .txt file (crucial for YOLO negative background learning).

    Args:
        df: DataFrame with stage_2_train_labels.csv columns.
        patient_split_map: Mapping of patientId -> 'train' | 'val' | 'test'.
        output_dir: Root dataset output directory.
        img_width: Native image width in pixels.
        img_height: Native image height in pixels.
        class_id: Class index for pneumonia_opacity (default: 0).

    Returns:
        Dict[str, int]: Count of label files generated per split.
    """
    out_path = Path(output_dir)
    labels_dir = out_path / "labels"

    for split in ["train", "val", "test"]:
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}

    # Group dataframe by patientId
    grouped = df.groupby("patientId")

    for patient_id, split in patient_split_map.items():
        label_file = labels_dir / split / f"{patient_id}.txt"
        lines: List[str] = []

        if patient_id in grouped.groups:
            p_rows = grouped.get_group(patient_id)
            for _, row in p_rows.iterrows():
                target = int(row["Target"])
                if target == 1 and pd.notna(row["x"]):
                    x = float(row["x"])
                    y = float(row["y"])
                    w = float(row["width"])
                    h = float(row["height"])

                    cls, xc, yc, nw, nh = convert_bbox_to_yolo(
                        x=x, y=y, width=w, height=h,
                        img_width=img_width, img_height=img_height,
                        class_id=class_id
                    )
                    lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        # Write lines (empty file if Target=0)
        label_file.write_text("\n".join(lines) + ("\n" if lines else ""))
        counts[split] += 1

    return counts


def generate_dataset_yaml(
    output_dir: Union[str, Path],
    class_name: str = CLASS_NAME_PNEUMONIA,
    has_test: bool = True
) -> Path:
    """Generate YOLO dataset.yaml configuration file.

    Args:
        output_dir: Root directory of YOLO dataset.
        class_name: Name of pneumonia opacity class.
        has_test: Whether to include 'test' path.

    Returns:
        Path: Path to written dataset.yaml.
    """
    out_path = Path(output_dir).resolve()

    yaml_data = {
        "path": str(out_path),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {
            0: class_name
        }
    }

    if has_test:
        yaml_data["test"] = "images/test"

    yaml_file = out_path / "dataset.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    return yaml_file

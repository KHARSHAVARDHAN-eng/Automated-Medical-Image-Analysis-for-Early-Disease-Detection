"""RSNA Pneumonia Detection label parsing and coordinate conversion utilities.

Translates between RSNA Challenge CSV annotations [x, y, width, height]
and normalized YOLO annotations [class_id, x_center, y_center, width, height].
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union
import pandas as pd


def convert_bbox_to_yolo(
    x: float,
    y: float,
    width: float,
    height: float,
    img_width: float = 1024.0,
    img_height: float = 1024.0,
    class_id: int = 0
) -> Tuple[int, float, float, float, float]:
    """Convert absolute [x, y, w, h] bounding box to normalized YOLO format.

    Args:
        x: Top-left x-coordinate.
        y: Top-left y-coordinate.
        width: Bounding box width.
        height: Bounding box height.
        img_width: Native image width (default: 1024).
        img_height: Native image height (default: 1024).
        class_id: Class integer index (0 for Lung Opacity).

    Returns:
        Tuple[int, float, float, float, float]:
        (class_id, x_center_norm, y_center_norm, width_norm, height_norm)
    """
    x_center = (x + width / 2.0) / img_width
    y_center = (y + height / 2.0) / img_height
    w_norm = width / img_width
    h_norm = height / img_height

    # Clip to valid [0.0, 1.0] range
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    return class_id, x_center, y_center, w_norm, h_norm


def convert_yolo_to_bbox(
    x_center: float,
    y_center: float,
    w_norm: float,
    h_norm: float,
    img_width: float = 1024.0,
    img_height: float = 1024.0
) -> Tuple[int, int, int, int]:
    """Convert normalized YOLO format back to pixel coordinates [x1, y1, x2, y2].

    Args:
        x_center: Normalized x-center (0.0 to 1.0).
        y_center: Normalized y-center (0.0 to 1.0).
        w_norm: Normalized width (0.0 to 1.0).
        h_norm: Normalized height (0.0 to 1.0).
        img_width: Target image width in pixels.
        img_height: Target image height in pixels.

    Returns:
        Tuple[int, int, int, int]: (x1, y1, x2, y2) corners in pixel coordinates.
    """
    abs_w = w_norm * img_width
    abs_h = h_norm * img_height
    abs_xc = x_center * img_width
    abs_yc = y_center * img_height

    x1 = int(round(abs_xc - abs_w / 2.0))
    y1 = int(round(abs_yc - abs_h / 2.0))
    x2 = int(round(abs_xc + abs_w / 2.0))
    y2 = int(round(abs_yc + abs_h / 2.0))

    return max(0, x1), max(0, y1), min(int(img_width), x2), min(int(img_height), y2)


def parse_ground_truth_labels(csv_path: Union[str, Path]) -> Dict[str, Dict]:
    """Parse RSNA stage_2_train_labels.csv into a structured dictionary.

    Groups multiple bounding boxes by patientId. Images without opacities
    (Target=0) have an empty boxes list.

    Args:
        csv_path: Path to stage_2_train_labels.csv.

    Returns:
        Dict[str, Dict]: Dictionary mapping patientId to:
        {
            "has_pneumonia": bool,
            "boxes": List[Tuple[float, float, float, float]]  # (x, y, w, h)
        }
    """
    df = pd.read_csv(csv_path)
    records: Dict[str, Dict] = {}

    for _, row in df.iterrows():
        patient_id = str(row["patientId"])
        target = int(row["Target"])

        if patient_id not in records:
            records[patient_id] = {
                "has_pneumonia": bool(target == 1),
                "boxes": []
            }

        if target == 1 and pd.notna(row["x"]):
            box = (float(row["x"]), float(row["y"]), float(row["width"]), float(row["height"]))
            records[patient_id]["boxes"].append(box)

    return records

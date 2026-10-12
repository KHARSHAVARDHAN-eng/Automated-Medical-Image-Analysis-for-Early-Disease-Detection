"""Preview RSNA chest radiographs with official ground-truth bounding box annotations.

Renders side-by-side comparisons of:
1. Original medical radiograph (processed via VOI LUT / MONOCHROME1 inversion).
2. Radiograph overlaid with official ground-truth bounding boxes and diagnostic label.
"""

import sys
from pathlib import Path
import cv2
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SAMPLES_DIR, RAW_DATA_DIR
from src.data.dicom_utils import read_dicom_image, get_dicom_metadata
from src.data.rsna_parser import parse_ground_truth_labels
from src.visualization.visualizer import overlay_ground_truth, create_side_by_side_comparison


def preview_samples():
    """Load samples from data/samples/ and generate ground-truth visualization previews."""
    dcm_files = sorted(list(SAMPLES_DIR.glob("*.dcm")))
    labels_csv = RAW_DATA_DIR / "stage_2_train_labels.csv"

    if not dcm_files:
        print(f"[INFO] No DICOM (.dcm) files found in {SAMPLES_DIR}.")
        print("Run `python3 scripts/download_rsna_samples.py` after accepting competition rules.")
        return

    ground_truth_records = {}
    if labels_csv.exists():
        print(f"Loading ground-truth annotations from: {labels_csv}")
        ground_truth_records = parse_ground_truth_labels(labels_csv)
    else:
        print(f"[WARNING] Labels CSV not found at {labels_csv}. Bounding boxes cannot be cross-referenced.")

    print(f"\nProcessing {len(dcm_files)} sample DICOM file(s):")
    print("=" * 80)

    for dcm_path in dcm_files:
        patient_id = dcm_path.stem

        # 1. Read DICOM with VOI LUT & Photometric Interpretation handling
        try:
            img = read_dicom_image(dcm_path)
            meta = get_dicom_metadata(dcm_path)
        except Exception as e:
            print(f"[ERROR] Failed to decode DICOM {dcm_path.name}: {e}")
            continue

        h, w = img.shape[:2]

        # 2. Extract ground truth
        record = ground_truth_records.get(patient_id, {"has_pneumonia": False, "boxes": []})
        has_pneumonia = record["has_pneumonia"]
        boxes = record["boxes"]

        status_str = "POSITIVE (Pneumonia)" if has_pneumonia else "NEGATIVE (Normal / No Opacity)"
        print(f"\nPatient ID       : {patient_id}")
        print(f"Dimensions       : {w} x {h}")
        print(f"Modality / View  : {meta.get('modality', 'CR')} / {meta.get('view_position', 'N/A')}")
        print(f"Age / Sex        : {meta.get('patient_age', 'N/A')} / {meta.get('patient_sex', 'N/A')}")
        print(f"Ground Truth     : {status_str}")
        print(f"Annotated Boxes  : {len(boxes)}")

        if boxes:
            for idx, box in enumerate(boxes, start=1):
                print(f"  Box #{idx}       : x={box[0]:.1f}, y={box[1]:.1f}, w={box[2]:.1f}, h={box[3]:.1f}")

        # 3. Create ground-truth overlay
        if boxes:
            annotated_img = overlay_ground_truth(img, boxes, label="Ground Truth: Lung Opacity")
        else:
            # If negative, draw green border indicator or label banner
            annotated_img = img.copy()
            cv2.putText(
                annotated_img,
                "Ground Truth: Normal (No Opacity)",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (50, 205, 50),
                2,
                cv2.LINE_AA,
            )

        # 4. Generate side-by-side comparison
        side_by_side = create_side_by_side_comparison(img, annotated_img)

        # 5. Save preview
        output_png = SAMPLES_DIR / f"preview_{patient_id}.png"
        cv2.imwrite(str(output_png), cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
        print(f"Saved Preview    : {output_png.name}")

    print("\n" + "=" * 80)
    print("Sample verification complete. Previews generated successfully.")


if __name__ == "__main__":
    preview_samples()

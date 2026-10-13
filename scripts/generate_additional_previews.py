"""Generate 30 additional chest X-ray ground-truth preview images for RSNA Pneumonia Detection.

Downloads 30 distinct RSNA DICOM cases (15 positive with pneumonia opacities and 15 negative normal)
without downloading the full 3.5GB dataset. Generates professional clinical side-by-side
ground-truth previews and writes an audit manifest CSV.

Does not overwrite existing previews.
Does not perform model training or inference.
Annotations represent radiologist-provided clinical ground truth.
"""

import sys
import os
import json
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR, SAMPLES_DIR, PREVIEWS_DIR
from src.data.dicom_utils import read_dicom_image, get_dicom_metadata
from src.data.rsna_parser import parse_ground_truth_labels
from src.visualization.visualizer import overlay_ground_truth, create_side_by_side_comparison

COMPETITION_ID = "rsna-pneumonia-detection-challenge"

# Initial 10 reference sample patient IDs (never duplicate these)
ORIGINAL_PREVIEW_PIDS = {
    "0004cfab-14fd-4e49-80ba-63a80b6bddd6",
    "00313ee0-9eaa-42f4-b0ab-c148ed3241cd",
    "00322d4d-1c29-4943-afc9-b6754be640eb",
    "003d8fa0-6bf1-40ed-b54c-ac657f8495c5",
    "00436515-870c-4b36-a041-de91049b9ab4",
    "00569f44-917d-4c86-a842-81832af98c30",
    "00704310-78a8-4b38-8475-49f4573b2dbb",
    "00aecb01-a116-45a2-956c-08d2fa55433f",
    "00c0b293-48e7-4e16-ac76-9269ba535a62",
    "00f08de1-517e-4652-a04f-d1dc9ee48593",
}

# The 30 curated new patient IDs (15 positive with 1, 2, or 3 boxes, 15 negative normal)
TARGET_30_PATIENTS = [
    # 15 Positive cases (with radiologist-annotated pneumonia opacity bounding boxes)
    {"patientId": "001916b8-3d30-4935-a5d1-8eaddb1646cd", "expected_target": 1, "expected_boxes": 1},
    {"patientId": "0100515c-5204-4f31-98e0-f35e4b00004a", "expected_target": 1, "expected_boxes": 1},
    {"patientId": "0101174b-6643-4d4e-b4ba-b6d41d0ce46a", "expected_target": 1, "expected_boxes": 1},
    {"patientId": "01adfd2f-7bc7-4cef-ab68-a0992752b620", "expected_target": 1, "expected_boxes": 1},
    {"patientId": "01cd2ba5-2baf-44b2-bf15-ee57e1ea4599", "expected_target": 1, "expected_boxes": 1},
    {"patientId": "01dc817c-3cfa-49fe-8662-518edc30652a", "expected_target": 1, "expected_boxes": 1},
    {"patientId": "02285fa4-35b7-4af6-b88f-3cac45a7f5c8", "expected_target": 1, "expected_boxes": 1},
    {"patientId": "000db696-cf54-4385-b10b-6b16fbb3f985", "expected_target": 1, "expected_boxes": 2},
    {"patientId": "000fe35a-2649-43d4-b027-e67796d412e0", "expected_target": 1, "expected_boxes": 2},
    {"patientId": "001031d9-f904-4a23-b3e5-2c088acd19c6", "expected_target": 1, "expected_boxes": 2},
    {"patientId": "0022073f-cec8-42ec-ab5f-bc2314649235", "expected_target": 1, "expected_boxes": 2},
    {"patientId": "002cb550-2e31-42f1-a29d-fbc279977e71", "expected_target": 1, "expected_boxes": 2},
    {"patientId": "0087bd3a-55a7-4045-b111-b018fa52d361", "expected_target": 1, "expected_boxes": 2},
    {"patientId": "02b7b018-0fed-44c2-a9c2-93b2b3952efc", "expected_target": 1, "expected_boxes": 3},
    {"patientId": "031d627f-e6af-4588-8cca-6be8aa1222c7", "expected_target": 1, "expected_boxes": 3},
    # 15 Negative cases (Normal chest X-rays with no annotated opacity boxes)
    {"patientId": "000924cf-0f8d-42bd-9158-1af53881a557", "expected_target": 0, "expected_boxes": 0},
    {"patientId": "0010f549-b242-4e94-87a8-57d79de215fc", "expected_target": 0, "expected_boxes": 0},
    {"patientId": "0022995a-45eb-4cfa-9a59-cd15f5196c64", "expected_target": 0, "expected_boxes": 0},
    {"patientId": "0025d2de-bd78-4d36-9f72-e15a5e22ca82", "expected_target": 0, "expected_boxes": 0},
    {"patientId": "00293de0-a530-41dc-9621-0b3def01d06d", "expected_target": 0, "expected_boxes": 0},
    {"patientId": "002c591d-df62-4e34-8eda-838c664430a9", "expected_target": 0, "expected_boxes": 0},
    {"patientId": "0049f7b2-229b-4a46-91a3-0e321a3ca80a", "expected_target": 0, "expected_boxes": 0},
    {"patientId": "0053ee32-1e03-4dc2-a103-f46ad0448787", "expected_target": 0, "expected_boxes": 0},
    {"patientId": "005f5517-ab76-4b0d-8a26-f808d97171c4", "expected_target": 0, "expected_boxes": 0},
    {"patientId": "0067d70b-0e6d-41ff-a9b8-8edf6d5e25e4", "expected_target": 0, "expected_boxes": 0},
    {"patientId": "006cec2e-6ce2-4549-bffa-eadfcd1e9970", "expected_target": 0, "expected_boxes": 0},
    {"patientId": "006e75c8-1fd9-4a5a-99e7-285addebed55", "expected_target": 0, "expected_boxes": 0},
    {"patientId": "0088be53-27f2-4c30-882b-a73a3a5c8c71", "expected_target": 0, "expected_boxes": 0},
    {"patientId": "008b69b2-446a-43dd-9ba2-9ccff8f3da41", "expected_target": 0, "expected_boxes": 0},
    {"patientId": "008c19e8-a820-403a-930a-bc74a4053664", "expected_target": 0, "expected_boxes": 0},
]


def add_header_banner(
    img: np.ndarray,
    title: str,
    subtitle: str,
    bg_color=(35, 39, 42),
    banner_height: int = 60
) -> np.ndarray:
    """Add a professional title banner above an image."""
    h, w = img.shape[:2]
    banner = np.zeros((banner_height, w, 3), dtype=np.uint8)
    banner[:] = bg_color

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Main title
    cv2.putText(banner, title, (20, 26), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    # Subtitle
    cv2.putText(banner, subtitle, (20, 48), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return np.vstack([banner, img])


def download_single_dicom(patient_id: str, dest_dir: Path) -> Path:
    """Download single DICOM file via Kaggle API if not present locally."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    dcm_filename = f"{patient_id}.dcm"
    target_dcm = dest_dir / dcm_filename

    if target_dcm.exists() and target_dcm.stat().st_size > 1000:
        return target_dcm

    api = KaggleApi()
    api.authenticate()

    comp_path = f"stage_2_train_images/{dcm_filename}"
    api.competition_download_file(
        competition=COMPETITION_ID,
        file_name=comp_path,
        path=str(dest_dir),
        force=False,
        quiet=True
    )

    zip_path = dest_dir / f"{dcm_filename}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest_dir)
        zip_path.unlink()

    if not target_dcm.exists():
        raise FileNotFoundError(f"Failed to acquire {dcm_filename}")

    return target_dcm


def generate_additional_previews(force_regenerate: bool = False):
    """Main workflow to download 30 new cases and generate ground truth previews."""
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    labels_csv = RAW_DATA_DIR / "stage_2_train_labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV missing at: {labels_csv}")

    # Load ground-truth records
    gt_records = parse_ground_truth_labels(labels_csv)

    print("=" * 80)
    print("RSNA PNEUMONIA GROUND-TRUTH PREVIEW GENERATOR")
    print("=" * 80)
    print(f"Target count: {len(TARGET_30_PATIENTS)} new previews (15 positive, 15 negative)")
    print(f"Output directory: {PREVIEWS_DIR}")

    # Step 1: Ensure DICOM files are present locally (download if missing)
    dcm_map = {}
    missing_pids = []
    for cand in TARGET_30_PATIENTS:
        pid = cand["patientId"]
        dcm_p = SAMPLES_DIR / f"{pid}.dcm"
        if dcm_p.exists() and dcm_p.stat().st_size > 1000:
            dcm_map[pid] = dcm_p
        else:
            missing_pids.append(pid)

    if missing_pids:
        print(f"\nDownloading {len(missing_pids)} missing DICOM files via Kaggle API...")
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_pid = {
                executor.submit(download_single_dicom, pid, SAMPLES_DIR): pid
                for pid in missing_pids
            }
            for future in as_completed(future_to_pid):
                pid = future_to_pid[future]
                dcm_p = future.result()
                dcm_map[pid] = dcm_p
                print(f" [OK] Downloaded: {dcm_p.name} ({dcm_p.stat().st_size:,} bytes)")
    else:
        print(f"All {len(TARGET_30_PATIENTS)} DICOM files are already cached locally in {SAMPLES_DIR}.")

    # Step 2: Generate previews for each patient
    manifest_records = []
    generated_count = 0
    skipped_count = 0

    print("\nProcessing and generating side-by-side clinical preview images:")
    for idx, cand in enumerate(TARGET_30_PATIENTS, start=1):
        pid = cand["patientId"]
        out_filename = f"preview_{pid}.png"
        out_path = PREVIEWS_DIR / out_filename

        rec = gt_records.get(pid, {"has_pneumonia": False, "boxes": []})
        has_pneumonia = rec["has_pneumonia"]
        boxes = rec["boxes"]
        status_label = (
            f"POSITIVE ({len(boxes)} Opacity Box{'es' if len(boxes) > 1 else ''})"
            if (has_pneumonia and boxes)
            else "NEGATIVE (Normal / No Opacity)"
        )

        # Check existing file
        if out_path.exists() and not force_regenerate:
            skipped_count += 1
            # Still verify metadata
            dcm_p = dcm_map[pid]
            meta = get_dicom_metadata(dcm_p)
            manifest_records.append({
                "patient_id": pid,
                "target": 1 if has_pneumonia else 0,
                "box_count": len(boxes),
                "status_label": status_label,
                "output_filename": out_filename,
                "output_path": str(out_path.relative_to(PROJECT_ROOT)),
                "dimensions": "1024x1024",
                "view_position": meta.get("view_position", "N/A"),
                "patient_age": meta.get("patient_age", "N/A"),
                "patient_sex": meta.get("patient_sex", "N/A"),
                "boxes_json": json.dumps(boxes),
            })
            print(f" [{idx:02d}/{len(TARGET_30_PATIENTS)}] [CACHED / SKIPPED] {out_filename} -> {status_label}")
            continue

        dcm_path = dcm_map[pid]
        img = read_dicom_image(dcm_path)
        meta = get_dicom_metadata(dcm_path)
        h, w = img.shape[:2]

        if has_pneumonia and boxes:
            annotated_img = overlay_ground_truth(img, boxes, label="Ground Truth: Lung Opacity")
            gt_banner_color = (25, 110, 40)  # Forest Green
            gt_title = f"GROUND TRUTH: POSITIVE | Opacity Boxes: {len(boxes)}"
        else:
            annotated_img = img.copy()
            cv2.putText(
                annotated_img,
                "Ground Truth: Normal (No Opacity Annotated)",
                (25, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (50, 205, 50),
                2,
                cv2.LINE_AA,
            )
            gt_banner_color = (30, 80, 110)  # Slate Teal
            gt_title = "GROUND TRUTH: NEGATIVE | Opacity Boxes: 0 (Normal)"

        view_pos = meta.get("view_position", "N/A")
        modality = meta.get("modality", "CR")
        orig_header = add_header_banner(
            img,
            title="ORIGINAL CHEST RADIOGRAPH",
            subtitle=f"Patient ID: {pid} | View: {view_pos} ({modality})",
            bg_color=(35, 39, 42)
        )
        annotated_header = add_header_banner(
            annotated_img,
            title=gt_title,
            subtitle="Official RSNA Clinical Radiologist Ground-Truth Annotation",
            bg_color=gt_banner_color
        )

        side_by_side = create_side_by_side_comparison(orig_header, annotated_header, divider_width=4)
        cv2.imwrite(str(out_path), cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))

        # Copy to SAMPLES_DIR for convenience
        sample_copy = SAMPLES_DIR / out_filename
        cv2.imwrite(str(sample_copy), cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))

        generated_count += 1
        manifest_records.append({
            "patient_id": pid,
            "target": 1 if has_pneumonia else 0,
            "box_count": len(boxes),
            "status_label": status_label,
            "output_filename": out_filename,
            "output_path": str(out_path.relative_to(PROJECT_ROOT)),
            "dimensions": f"{w}x{h}",
            "view_position": view_pos,
            "patient_age": meta.get("patient_age", "N/A"),
            "patient_sex": meta.get("patient_sex", "N/A"),
            "boxes_json": json.dumps(boxes),
        })
        print(f" [{idx:02d}/{len(TARGET_30_PATIENTS)}] [GENERATED] {out_filename} -> {status_label}")

    # Step 3: Write manifests
    manifest_df = pd.DataFrame(manifest_records)
    manifest_path = PREVIEWS_DIR / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\n[MANIFEST] Saved manifest for 30 new previews to: {manifest_path}")

    # Comprehensive folder manifest (original 10 + 30 new = 40 total)
    all_previews = sorted(list(PREVIEWS_DIR.glob("preview_*.png")))
    complete_records = []
    for p_file in all_previews:
        pid = p_file.stem.replace("preview_", "")
        rec = gt_records.get(pid, {"has_pneumonia": False, "boxes": []})
        complete_records.append({
            "patient_id": pid,
            "target": 1 if rec["has_pneumonia"] else 0,
            "box_count": len(rec["boxes"]),
            "output_filename": p_file.name,
            "output_path": str(p_file.relative_to(PROJECT_ROOT)),
        })
    complete_df = pd.DataFrame(complete_records)
    complete_path = PREVIEWS_DIR / "complete_previews_manifest.csv"
    complete_df.to_csv(complete_path, index=False)
    print(f"[MANIFEST] Saved complete folder manifest to: {complete_path} ({len(complete_records)} total entries)")

    # Step 4: Verification of all 30 files
    print("\nVerifying that all 30 target preview images exist and can be opened...")
    for rec in manifest_records:
        fpath = PREVIEWS_DIR / rec["output_filename"]
        assert fpath.exists(), f"Missing file: {fpath}"
        assert fpath.stat().st_size > 10000, f"File unexpectedly small: {fpath}"
        loaded = cv2.imread(str(fpath))
        assert loaded is not None, f"Failed to open image: {fpath}"
        assert loaded.shape == (1084, 2052, 3), f"Unexpected shape {loaded.shape} in {fpath}"

    print(f"[VERIFIED] All 30 preview files verified (1084x2052 RGB side-by-side)!")
    print(f"Total new previews generated: {generated_count}")
    print(f"Pre-existing previews skipped: {skipped_count}")
    print(f"Total previews available in {PREVIEWS_DIR}: {len(all_previews)}")

    return generated_count, skipped_count, PREVIEWS_DIR


if __name__ == "__main__":
    gen, skip, pdir = generate_additional_previews()
    print(f"\nSummary: {gen} generated, {skip} skipped in {pdir}")

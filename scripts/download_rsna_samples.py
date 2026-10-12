"""Download official RSNA Pneumonia Detection Challenge labels and sample DICOMs.

Selectively downloads stage_2_train_labels.csv and exactly 2 real DICOM chest radiographs:
- One positive sample (with annotated pneumonia-related lung opacity bounding boxes).
- One negative sample (normal / no lung opacity).

Requires accepted competition rules at:
https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/rules
"""

import sys
import os
import zipfile
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR, SAMPLES_DIR

COMPETITION_ID = "rsna-pneumonia-detection-challenge"
RULES_URL = f"https://www.kaggle.com/competitions/{COMPETITION_ID}/rules"


def check_kaggle_auth():
    """Verify Kaggle API credentials existence."""
    creds_path = Path.home() / ".kaggle" / "kaggle.json"
    env_creds = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    if not creds_path.exists() and not env_creds:
        print("[ERROR] Kaggle credentials not found!")
        print("Please place your kaggle.json file in ~/.kaggle/kaggle.json")
        return False
    return True


def download_official_samples():
    """Download labels CSV and 2 matching authentic RSNA DICOM images."""
    if not check_kaggle_auth():
        sys.exit(1)

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print(f"[ERROR] Kaggle API authentication failed: {e}")
        sys.exit(1)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    labels_csv = RAW_DATA_DIR / "stage_2_train_labels.csv"

    # Step 1: Download stage_2_train_labels.csv
    if not labels_csv.exists():
        print(f"Downloading stage_2_train_labels.csv from competition '{COMPETITION_ID}'...")
        try:
            api.competition_download_file(
                competition=COMPETITION_ID,
                file_name="stage_2_train_labels.csv",
                path=str(RAW_DATA_DIR),
                force=False,
                quiet=False
            )
            # Unzip if downloaded as zip
            zip_target = RAW_DATA_DIR / "stage_2_train_labels.csv.zip"
            if zip_target.exists():
                with zipfile.ZipFile(zip_target, "r") as z:
                    z.extractall(RAW_DATA_DIR)
                zip_target.unlink()
            print(f"[SUCCESS] Downloaded: {labels_csv}")
        except Exception as e:
            err_str = str(e)
            if "403" in err_str or "rules" in err_str.lower():
                print("\n" + "=" * 70)
                print("[KAGGLE COMPETITION RULES REQUIRED]")
                print(f"You must accept the competition rules in your Kaggle browser account:")
                print(f" -> {RULES_URL}")
                print("1. Open the URL above in your browser.")
                print("2. Click 'I Understand and Accept'.")
                print("3. Re-run this script.")
                print("=" * 70 + "\n")
                sys.exit(2)
            else:
                print(f"[ERROR] Failed to download labels CSV: {e}")
                sys.exit(1)
    else:
        print(f"[CACHE] Using existing labels CSV: {labels_csv}")

    # Step 2: Parse CSV and identify 1 positive and 1 negative sample
    df = pd.read_csv(labels_csv)
    positives = df[df["Target"] == 1]
    negatives = df[df["Target"] == 0]

    # Select representative samples
    # 00436515-870c-4b36-a041-de91049b9ab4 has 2 distinct opacities
    pos_id = "00436515-870c-4b36-a041-de91049b9ab4"
    if pos_id not in positives["patientId"].values:
        pos_id = positives.iloc[0]["patientId"]

    # 0004cfab-14fd-4e49-80ba-63a80b6bddd6 is negative (Target = 0)
    neg_id = "0004cfab-14fd-4e49-80ba-63a80b6bddd6"
    if neg_id not in negatives["patientId"].values:
        neg_id = negatives.iloc[0]["patientId"]

    samples_to_fetch = [
        (pos_id, "positive"),
        (neg_id, "negative")
    ]

    print(f"\nTarget samples identified from official labels CSV:")
    print(f" - Positive sample ID : {pos_id} (Target=1)")
    print(f" - Negative sample ID : {neg_id} (Target=0)")

    # Step 3: Selectively download individual DICOM files
    for patient_id, label_type in samples_to_fetch:
        dcm_filename = f"{patient_id}.dcm"
        local_dcm_path = SAMPLES_DIR / dcm_filename

        if local_dcm_path.exists():
            print(f"[CACHE] {label_type.capitalize()} DICOM already present: {local_dcm_path.name}")
            continue

        comp_file_path = f"stage_2_train_images/{dcm_filename}"
        print(f"Downloading {label_type} DICOM: {comp_file_path}...")
        try:
            api.competition_download_file(
                competition=COMPETITION_ID,
                file_name=comp_file_path,
                path=str(SAMPLES_DIR),
                force=False,
                quiet=False
            )
            # Unpack if zip
            zip_path = SAMPLES_DIR / f"{dcm_filename}.zip"
            if zip_path.exists():
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(SAMPLES_DIR)
                zip_path.unlink()
            print(f"[SUCCESS] Downloaded: {local_dcm_path.name}")
        except Exception as e:
            print(f"[ERROR] Failed to download {dcm_filename}: {e}")

    print("\nDownload process completed.")


if __name__ == "__main__":
    download_official_samples()

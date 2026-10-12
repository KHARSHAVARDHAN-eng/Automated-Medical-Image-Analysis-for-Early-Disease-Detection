"""Prepare and validate YOLO formatted dataset from RSNA labels.

Performs patient-level stratified splitting, converts annotations into normalized
YOLO format (with empty label files for negative samples), generates dataset.yaml,
and runs comprehensive integrity validation.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CLASS_NAME_PNEUMONIA
from src.data.dataset_preparer import split_patients, generate_yolo_labels, generate_dataset_yaml
from src.data.dataset_validator import validate_splits_leakage, validate_yolo_dataset


def run_preparation_and_validation(
    labels_csv_path: Path = RAW_DATA_DIR / "stage_2_train_labels.csv",
    output_dir: Path = PROCESSED_DATA_DIR / "rsna_yolo",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42
):
    """Execute patient-level split, label generation, and integrity check."""
    print("=" * 80)
    print("RSNA PNEUMONIA DETECTION: YOLO DATASET PREPARATION & VALIDATION")
    print("=" * 80)

    if not labels_csv_path.exists():
        print(f"[ERROR] Labels CSV not found: {labels_csv_path}")
        return False

    print(f"Reading labels from: {labels_csv_path}")
    df = pd.read_csv(labels_csv_path)
    total_rows = len(df)
    unique_patients = df["patientId"].nunique()
    positives = df[df["Target"] == 1]["patientId"].nunique()
    negatives = df[df["Target"] == 0]["patientId"].nunique()

    print(f"Total Annotation Rows : {total_rows:,}")
    print(f"Unique Patient IDs    : {unique_patients:,}")
    print(f" - Positive Patients  : {positives:,} ({positives / unique_patients:.1%})")
    print(f" - Negative Patients  : {negatives:,} ({negatives / unique_patients:.1%})")

    # 1. Patient-level stratified split
    print(f"\nPerforming patient-level stratified split (Train: {train_ratio:.0%}, Val: {val_ratio:.0%}, Test: {test_ratio:.0%})...")
    patient_split_map = split_patients(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )

    # 2. Validate zero patient leakage
    leakage_check = validate_splits_leakage(patient_split_map)
    if not leakage_check["is_valid"]:
        print(f"[FAIL] Patient leakage detected: {leakage_check['errors']}")
        return False
    print(f"[PASS] Zero patient data leakage confirmed across all splits.")
    for split_name, count in leakage_check["split_counts"].items():
        print(f" - {split_name.capitalize():<5} Patients: {count:,}")

    # 3. Generate YOLO labels
    print(f"\nGenerating YOLO labels in: {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = generate_yolo_labels(
        df=df,
        patient_split_map=patient_split_map,
        output_dir=output_dir,
        class_id=0
    )
    for split_name, count in counts.items():
        print(f" - Generated {count:,} label files in labels/{split_name}/")

    # 4. Generate dataset.yaml
    yaml_path = generate_dataset_yaml(output_dir, class_name=CLASS_NAME_PNEUMONIA)
    print(f"[SUCCESS] Dataset YAML written to: {yaml_path}")

    # 5. Run full dataset validation
    print("\nRunning comprehensive dataset validation check...")
    val_results = validate_yolo_dataset(output_dir, require_images=False)
    if not val_results["is_valid"]:
        print(f"[FAIL] Dataset validation failed with errors: {val_results['errors']}")
        return False

    print("[PASS] Dataset structure and label formats 100% valid!")
    print("\nSplit Summary Statistics:")
    print("-" * 75)
    print(f"{'Split':<8} | {'Total Samples':<14} | {'Positives':<10} | {'Negatives':<10} | {'Total Boxes':<12}")
    print("-" * 75)
    for s_name, s_data in val_results["splits"].items():
        print(
            f"{s_name.capitalize():<8} | "
            f"{s_data['total_samples']:<14,}"
            f" | {s_data['positive_samples']:<10,}"
            f" | {s_data['negative_samples']:<10,}"
            f" | {s_data['total_boxes']:<12,}"
        )
    print("-" * 75)
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = run_preparation_and_validation()
    if not success:
        sys.exit(1)

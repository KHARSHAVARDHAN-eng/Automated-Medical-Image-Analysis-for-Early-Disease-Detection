"""Integration tests for downloaded real RSNA samples and preview outputs."""

import unittest
from pathlib import Path
import cv2
import pandas as pd

from src.config import PROJECT_ROOT, RAW_DATA_DIR, SAMPLES_DIR, PREVIEWS_DIR, REPORTS_DIR
from src.data.dicom_utils import read_dicom_image, get_dicom_metadata
from src.data.rsna_parser import parse_ground_truth_labels


# 10 verified samples in project
VERIFIED_SAMPLES = [
    {"patient_id": "00436515-870c-4b36-a041-de91049b9ab4", "target": 1, "expected_boxes": 2},
    {"patient_id": "00704310-78a8-4b38-8475-49f4573b2dbb", "target": 1, "expected_boxes": 2},
    {"patient_id": "00aecb01-a116-45a2-956c-08d2fa55433f", "target": 1, "expected_boxes": 2},
    {"patient_id": "00c0b293-48e7-4e16-ac76-9269ba535a62", "target": 1, "expected_boxes": 2},
    {"patient_id": "00f08de1-517e-4652-a04f-d1dc9ee48593", "target": 1, "expected_boxes": 2},
    {"patient_id": "0004cfab-14fd-4e49-80ba-63a80b6bddd6", "target": 0, "expected_boxes": 0},
    {"patient_id": "00313ee0-9eaa-42f4-b0ab-c148ed3241cd", "target": 0, "expected_boxes": 0},
    {"patient_id": "00322d4d-1c29-4943-afc9-b6754be640eb", "target": 0, "expected_boxes": 0},
    {"patient_id": "003d8fa0-6bf1-40ed-b54c-ac657f8495c5", "target": 0, "expected_boxes": 0},
    {"patient_id": "00569f44-917d-4c86-a842-81832af98c30", "target": 0, "expected_boxes": 0},
]


class TestSampleIntegration(unittest.TestCase):
    """Verify downloaded official RSNA samples and preview generations."""

    def test_labels_csv_integrity(self):
        """Verify downloaded stage_2_train_labels.csv structure and sample presence."""
        csv_path = RAW_DATA_DIR / "stage_2_train_labels.csv"
        self.assertTrue(csv_path.exists(), "stage_2_train_labels.csv must be present.")

        df = pd.read_csv(csv_path)
        expected_cols = {"patientId", "x", "y", "width", "height", "Target"}
        self.assertTrue(expected_cols.issubset(set(df.columns)), f"Columns missing from {df.columns}")
        self.assertGreater(len(df), 30000, "Dataset should have over 30,000 annotation rows.")

        # Verify all 10 samples in the CSV
        for s in VERIFIED_SAMPLES:
            pid = s["patient_id"]
            rows = df[df["patientId"] == pid]
            self.assertGreater(len(rows), 0, f"Patient {pid} must exist in CSV.")
            if s["target"] == 1:
                self.assertEqual(len(rows), s["expected_boxes"], f"Positive sample {pid} must have {s['expected_boxes']} box rows.")
                self.assertTrue((rows["Target"] == 1).all(), f"All rows for {pid} must have Target=1.")
            else:
                self.assertEqual(len(rows), 1, f"Negative sample {pid} must have 1 row.")
                self.assertEqual(int(rows.iloc[0]["Target"]), 0, f"Negative sample {pid} must have Target=0.")

    def test_dicom_samples_decoding(self):
        """Verify all 10 genuine downloaded DICOM samples decode to valid (1024, 1024, 3) images."""
        for s in VERIFIED_SAMPLES:
            pid = s["patient_id"]
            dcm_path = SAMPLES_DIR / f"{pid}.dcm"
            self.assertTrue(dcm_path.exists(), f"DICOM file for {pid} must exist.")

            img = read_dicom_image(dcm_path)
            self.assertEqual(img.shape, (1024, 1024, 3), f"Image {pid} shape should be (1024, 1024, 3).")
            meta = get_dicom_metadata(dcm_path)
            self.assertEqual(meta["patient_id"], pid, f"Metadata patient_id should match {pid}.")

    def test_all_10_previews_generated(self):
        """Verify all 10 preview PNGs are present and readable with expected composite dimensions."""
        for s in VERIFIED_SAMPLES:
            pid = s["patient_id"]
            preview_path = PREVIEWS_DIR / f"preview_{pid}.png"
            self.assertTrue(preview_path.exists(), f"Preview PNG for {pid} must exist.")

            img = cv2.imread(str(preview_path))
            self.assertIsNotNone(img, f"Preview image {pid} must be readable by OpenCV.")
            # 1024 + 60 (header) = 1084 height; 1024 + 4 + 1024 = 2052 width
            self.assertEqual(img.shape[0], 1084, f"Height of {pid} should be 1084.")
            self.assertEqual(img.shape[1], 2052, f"Width of {pid} should be 2052.")

    def test_demo_html_and_py_reference_all_10_samples(self):
        """Verify demo.html and demo.py reference all 10 verified patient IDs."""
        html_file = PROJECT_ROOT / "demo.html"
        py_file = PROJECT_ROOT / "demo.py"

        self.assertTrue(html_file.exists(), "demo.html must exist.")
        self.assertTrue(py_file.exists(), "demo.py must exist.")

        html_content = html_file.read_text()
        py_content = py_file.read_text()

        for s in VERIFIED_SAMPLES:
            pid = s["patient_id"]
            self.assertIn(pid, html_content, f"demo.html must contain patient ID {pid}")
            self.assertIn(pid, py_content, f"demo.py must contain patient ID {pid}")


if __name__ == "__main__":
    unittest.main()

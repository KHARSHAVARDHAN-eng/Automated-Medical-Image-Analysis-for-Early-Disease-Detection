"""Integration tests for downloaded real RSNA samples and preview outputs."""

import unittest
from pathlib import Path
import cv2
import pandas as pd

from src.config import RAW_DATA_DIR, SAMPLES_DIR, PREVIEWS_DIR, REPORTS_DIR
from src.data.dicom_utils import read_dicom_image, get_dicom_metadata
from src.data.rsna_parser import parse_ground_truth_labels


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

        # Verify positive sample is present
        pos_id = "00436515-870c-4b36-a041-de91049b9ab4"
        pos_rows = df[df["patientId"] == pos_id]
        self.assertEqual(len(pos_rows), 2, "Positive sample must have exactly 2 opacity rows.")
        self.assertTrue((pos_rows["Target"] == 1).all(), "All rows for positive sample must have Target=1.")

        # Verify negative sample is present
        neg_id = "0004cfab-14fd-4e49-80ba-63a80b6bddd6"
        neg_rows = df[df["patientId"] == neg_id]
        self.assertEqual(len(neg_rows), 1, "Negative sample must have 1 row.")
        self.assertEqual(int(neg_rows.iloc[0]["Target"]), 0, "Negative sample must have Target=0.")

    def test_dicom_samples_decoding(self):
        """Verify genuine downloaded DICOM samples decode to valid (1024, 1024, 3) images."""
        pos_dcm = SAMPLES_DIR / "00436515-870c-4b36-a041-de91049b9ab4.dcm"
        neg_dcm = SAMPLES_DIR / "0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm"

        self.assertTrue(pos_dcm.exists(), "Positive sample DICOM must exist.")
        self.assertTrue(neg_dcm.exists(), "Negative sample DICOM must exist.")

        # Test positive DICOM
        pos_img = read_dicom_image(pos_dcm)
        self.assertEqual(pos_img.shape, (1024, 1024, 3))
        pos_meta = get_dicom_metadata(pos_dcm)
        self.assertEqual(pos_meta["patient_id"], "00436515-870c-4b36-a041-de91049b9ab4")

        # Test negative DICOM
        neg_img = read_dicom_image(neg_dcm)
        self.assertEqual(neg_img.shape, (1024, 1024, 3))
        neg_meta = get_dicom_metadata(neg_dcm)
        self.assertEqual(neg_meta["patient_id"], "0004cfab-14fd-4e49-80ba-63a80b6bddd6")

    def test_previews_and_reports_generated(self):
        """Verify preview PNGs and report are present and readable."""
        preview_pos = PREVIEWS_DIR / "preview_00436515-870c-4b36-a041-de91049b9ab4.png"
        preview_neg = PREVIEWS_DIR / "preview_0004cfab-14fd-4e49-80ba-63a80b6bddd6.png"
        report_file = REPORTS_DIR / "dataset_verification_report.md"

        self.assertTrue(preview_pos.exists(), "Positive preview PNG must exist.")
        self.assertTrue(preview_neg.exists(), "Negative preview PNG must exist.")
        self.assertTrue(report_file.exists(), "Dataset verification report must exist.")

        # Verify images can be opened and have combined side-by-side dimensions
        img_pos = cv2.imread(str(preview_pos))
        self.assertIsNotNone(img_pos)
        # 1024 + 60 (header) = 1084 height; 1024 + 4 + 1024 = 2052 width
        self.assertEqual(img_pos.shape[0], 1084)
        self.assertEqual(img_pos.shape[1], 2052)


if __name__ == "__main__":
    unittest.main()

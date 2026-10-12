"""Unit tests for RSNA annotation CSV parsing and multi-box grouping."""

import unittest
import tempfile
from pathlib import Path
import pandas as pd

from src.data.rsna_parser import parse_ground_truth_labels


class TestAnnotationParser(unittest.TestCase):
    """Test suite for RSNA CSV label ingestion."""

    def setUp(self):
        """Create a mock stage_2_train_labels.csv."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "mock_labels.csv"

        # Mock RSNA data:
        # Patient A: 2 lung opacities (pneumonia)
        # Patient B: 0 opacities (normal/negative)
        # Patient C: 1 opacity (pneumonia)
        df = pd.DataFrame([
            {"patientId": "PATIENT_A", "x": 200.0, "y": 300.0, "width": 100.0, "height": 150.0, "Target": 1},
            {"patientId": "PATIENT_A", "x": 550.0, "y": 320.0, "width": 120.0, "height": 160.0, "Target": 1},
            {"patientId": "PATIENT_B", "x": float("nan"), "y": float("nan"), "width": float("nan"), "height": float("nan"), "Target": 0},
            {"patientId": "PATIENT_C", "x": 180.0, "y": 250.0, "width": 140.0, "height": 200.0, "Target": 1},
        ])
        df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_multi_box_positive_patient(self):
        """Verify grouping of multiple bounding boxes for the same patient."""
        records = parse_ground_truth_labels(self.csv_path)

        self.assertIn("PATIENT_A", records)
        self.assertTrue(records["PATIENT_A"]["has_pneumonia"])
        self.assertEqual(len(records["PATIENT_A"]["boxes"]), 2)
        self.assertEqual(records["PATIENT_A"]["boxes"][0], (200.0, 300.0, 100.0, 150.0))
        self.assertEqual(records["PATIENT_A"]["boxes"][1], (550.0, 320.0, 120.0, 160.0))

    def test_negative_patient_empty_boxes(self):
        """Verify Target=0 patients have has_pneumonia=False and empty box list."""
        records = parse_ground_truth_labels(self.csv_path)

        self.assertIn("PATIENT_B", records)
        self.assertFalse(records["PATIENT_B"]["has_pneumonia"])
        self.assertEqual(len(records["PATIENT_B"]["boxes"]), 0)

    def test_single_box_patient(self):
        """Verify single box patient."""
        records = parse_ground_truth_labels(self.csv_path)

        self.assertIn("PATIENT_C", records)
        self.assertTrue(records["PATIENT_C"]["has_pneumonia"])
        self.assertEqual(len(records["PATIENT_C"]["boxes"]), 1)

    def test_missing_csv_file(self):
        """Verify error when CSV does not exist."""
        non_existent = Path(self.temp_dir.name) / "missing.csv"
        with self.assertRaises(Exception):
            parse_ground_truth_labels(non_existent)


if __name__ == "__main__":
    unittest.main()

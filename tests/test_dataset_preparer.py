"""Unit tests for patient-level splitting, YOLO formatting, and dataset.yaml generation."""

import unittest
import tempfile
from pathlib import Path
import pandas as pd
import yaml

from src.data.dataset_preparer import split_patients, generate_yolo_labels, generate_dataset_yaml
from src.config import CLASS_NAME_PNEUMONIA


class TestDatasetPreparer(unittest.TestCase):
    """Test suite for dataset splitting and YOLO format translation."""

    def setUp(self):
        """Build mock multi-patient dataframe."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.temp_dir.name)

        # 10 patients: 4 positive, 6 negative
        # Patient P0 has 2 boxes
        rows = [
            {"patientId": "P0", "x": 100.0, "y": 150.0, "width": 80.0, "height": 90.0, "Target": 1},
            {"patientId": "P0", "x": 400.0, "y": 300.0, "width": 110.0, "height": 120.0, "Target": 1},
            {"patientId": "P1", "x": 200.0, "y": 250.0, "width": 90.0, "height": 100.0, "Target": 1},
            {"patientId": "P2", "x": 300.0, "y": 350.0, "width": 70.0, "height": 80.0, "Target": 1},
            {"patientId": "P3", "x": 150.0, "y": 180.0, "width": 95.0, "height": 105.0, "Target": 1},
            {"patientId": "N0", "x": float("nan"), "y": float("nan"), "width": float("nan"), "height": float("nan"), "Target": 0},
            {"patientId": "N1", "x": float("nan"), "y": float("nan"), "width": float("nan"), "height": float("nan"), "Target": 0},
            {"patientId": "N2", "x": float("nan"), "y": float("nan"), "width": float("nan"), "height": float("nan"), "Target": 0},
            {"patientId": "N3", "x": float("nan"), "y": float("nan"), "width": float("nan"), "height": float("nan"), "Target": 0},
            {"patientId": "N4", "x": float("nan"), "y": float("nan"), "width": float("nan"), "height": float("nan"), "Target": 0},
            {"patientId": "N5", "x": float("nan"), "y": float("nan"), "width": float("nan"), "height": float("nan"), "Target": 0},
        ]
        self.df = pd.DataFrame(rows)

    def tearDown(self):
        """Cleanup temporary directory."""
        self.temp_dir.cleanup()

    def test_patient_level_splitting_no_leakage(self):
        """Verify patient-level split contains zero overlapping patients."""
        split_map = split_patients(self.df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42)

        train_pts = {pid for pid, s in split_map.items() if s == "train"}
        val_pts = {pid for pid, s in split_map.items() if s == "val"}
        test_pts = {pid for pid, s in split_map.items() if s == "test"}

        # Total patients accounted for
        self.assertEqual(len(train_pts) + len(val_pts) + len(test_pts), 10)

        # Mutually disjoint sets
        self.assertEqual(len(train_pts.intersection(val_pts)), 0, "Leakage between train and val!")
        self.assertEqual(len(train_pts.intersection(test_pts)), 0, "Leakage between train and test!")
        self.assertEqual(len(val_pts.intersection(test_pts)), 0, "Leakage between val and test!")

    def test_yolo_labels_generation_positive_and_negative(self):
        """Verify label file contents for multi-box positive and empty negative patients."""
        split_map = {"P0": "train", "N0": "val"}
        counts = generate_yolo_labels(self.df, split_map, self.out_dir, class_id=0)

        self.assertEqual(counts["train"], 1)
        self.assertEqual(counts["val"], 1)

        # Positive P0 should have 2 lines
        p0_file = self.out_dir / "labels" / "train" / "P0.txt"
        self.assertTrue(p0_file.exists())
        lines = [line.strip() for line in p0_file.read_text().splitlines() if line.strip()]
        self.assertEqual(len(lines), 2)
        # Check first line starts with class 0
        self.assertTrue(lines[0].startswith("0 "))
        self.assertTrue(lines[1].startswith("0 "))

        # Negative N0 should be an empty file (0 lines)
        n0_file = self.out_dir / "labels" / "val" / "N0.txt"
        self.assertTrue(n0_file.exists())
        n0_content = n0_file.read_text().strip()
        self.assertEqual(n0_content, "", "Negative label file must be empty!")

    def test_dataset_yaml_structure(self):
        """Verify dataset.yaml is formatted correctly with class name."""
        yaml_path = generate_dataset_yaml(self.out_dir, class_name=CLASS_NAME_PNEUMONIA)
        self.assertTrue(yaml_path.exists())

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        self.assertEqual(data["nc"], 1)
        self.assertEqual(data["names"][0], "pneumonia_opacity")
        self.assertIn("train", data)
        self.assertIn("val", data)
        self.assertIn("test", data)


if __name__ == "__main__":
    unittest.main()

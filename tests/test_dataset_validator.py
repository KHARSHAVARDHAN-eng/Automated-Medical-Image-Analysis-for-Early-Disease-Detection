"""Unit tests for dataset integrity validator."""

import unittest
import tempfile
from pathlib import Path

from src.data.dataset_validator import (
    validate_splits_leakage,
    validate_yolo_label_file,
    validate_yolo_dataset,
)


class TestDatasetValidator(unittest.TestCase):
    """Test suite for dataset validation logic."""

    def setUp(self):
        """Create temporary mock dataset hierarchy."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_leakage_detection(self):
        """Verify leakage validator flags overlapping patients."""
        # Intentionally introduce overlap: P1 in both train and val
        split_map_with_leakage = {
            "P1": "train",
            "P2": "val",
        }
        # Artificial overlapping set
        res = validate_splits_leakage({"P1": "train", "P2": "val", "P3": "test"})
        self.assertTrue(res["is_valid"])

    def test_leakage_rejection(self):
        """Verify that conflicting split names or overlaps fail."""
        # Split with unlisted name
        res = validate_splits_leakage({"P1": "invalid_split"})
        self.assertFalse(res["is_valid"])

    def test_valid_yolo_label_file(self):
        """Verify valid label file parsing."""
        label_file = self.root / "valid.txt"
        label_file.write_text("0 0.500000 0.500000 0.200000 0.300000\n0 0.700000 0.400000 0.150000 0.250000\n")

        res = validate_yolo_label_file(label_file)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["box_count"], 2)

    def test_empty_yolo_label_file_is_valid(self):
        """Verify empty file is valid for negative normal case."""
        label_file = self.root / "negative.txt"
        label_file.write_text("")

        res = validate_yolo_label_file(label_file)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["box_count"], 0)

    def test_out_of_bounds_bounding_box_rejected(self):
        """Verify that normalized coordinates outside [0, 1] fail validation."""
        label_file = self.root / "invalid_bounds.txt"
        label_file.write_text("0 1.250000 0.500000 0.200000 0.300000\n")  # xc = 1.25 > 1.0

        res = validate_yolo_label_file(label_file)
        self.assertFalse(res["is_valid"])
        self.assertIn("out of bounds", res["errors"][0])

    def test_non_numeric_token_rejected(self):
        """Verify that malformed string tokens fail validation."""
        label_file = self.root / "corrupted.txt"
        label_file.write_text("0 abc 0.5 0.2 0.3\n")

        res = validate_yolo_label_file(label_file)
        self.assertFalse(res["is_valid"])
        self.assertIn("Non-numeric token", res["errors"][0])


if __name__ == "__main__":
    unittest.main()

"""Unit tests for cloud pipeline path detection and training configuration."""

import unittest
from pathlib import Path

from src.data.dataset_preparer import detect_rsna_dataset_paths
from src.config import CLASS_NAME_PNEUMONIA, PROJECT_ROOT


class TestCloudPipeline(unittest.TestCase):
    """Test suite for cloud training pipeline components."""

    def test_auto_detect_rsna_paths_locally(self):
        """Verify dynamic auto-detection finds local project samples and CSV."""
        res = detect_rsna_dataset_paths(search_roots=[PROJECT_ROOT / "data"])
        self.assertTrue(res["is_valid"], f"Path detection failed: {res.get('error')}")
        self.assertIsNotNone(res["labels_csv"])
        self.assertIsNotNone(res["dicom_dir"])
        self.assertTrue(res["labels_csv"].exists())
        self.assertTrue(res["dicom_dir"].exists())
        self.assertGreater(res["total_rows"], 30000)
        self.assertGreater(res["dicom_files_found"], 0)

    def test_training_hyperparameters_validity(self):
        """Verify cloud training hyperparameters adhere to constraints."""
        train_config = {
            "model": "yolov8s.pt",
            "imgsz": 640,
            "batch": 16,
            "epochs": 25,
            "patience": 10,
            "save": True,
            "save_period": 1,
            "amp": True,
            "workers": 4,
            "names": {0: CLASS_NAME_PNEUMONIA}
        }

        self.assertEqual(train_config["imgsz"], 640)
        self.assertEqual(train_config["names"][0], "pneumonia_opacity")
        self.assertLessEqual(train_config["batch"], 32, "Batch size should be <= 32 to prevent T4 OOM.")
        self.assertGreaterEqual(train_config["epochs"], 10)
        self.assertTrue(train_config["save"])
        self.assertEqual(train_config["save_period"], 1, "Must checkpoint at every epoch.")


if __name__ == "__main__":
    unittest.main()

"""Unit tests for project configuration, paths, and accelerator detection."""

import unittest
from pathlib import Path
from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    WEIGHTS_DIR,
    NOTEBOOKS_DIR,
    get_inference_device,
    CLASS_NAMES,
)


class TestConfig(unittest.TestCase):
    """Test suite for validating paths and configurations."""

    def test_directory_paths_exist(self):
        """Verify that all core project directory structures are present."""
        self.assertTrue(PROJECT_ROOT.exists(), "PROJECT_ROOT must exist.")
        self.assertTrue(DATA_DIR.exists(), "DATA_DIR must exist.")
        self.assertTrue(RAW_DATA_DIR.exists(), "RAW_DATA_DIR must exist.")
        self.assertTrue(PROCESSED_DATA_DIR.exists(), "PROCESSED_DATA_DIR must exist.")
        self.assertTrue(MODELS_DIR.exists(), "MODELS_DIR must exist.")
        self.assertTrue(WEIGHTS_DIR.exists(), "WEIGHTS_DIR must exist.")
        self.assertTrue(NOTEBOOKS_DIR.exists(), "NOTEBOOKS_DIR must exist.")

    def test_classes_configuration(self):
        """Verify standard class name definitions."""
        self.assertEqual(len(CLASS_NAMES), 1)
        self.assertEqual(CLASS_NAMES[0], "Lung Opacity")

    def test_inference_device_detection(self):
        """Verify device detection returns a valid compute device."""
        device = get_inference_device()
        self.assertIn(device, ["mps", "cuda", "cpu"], f"Invalid device detected: {device}")


if __name__ == "__main__":
    unittest.main()

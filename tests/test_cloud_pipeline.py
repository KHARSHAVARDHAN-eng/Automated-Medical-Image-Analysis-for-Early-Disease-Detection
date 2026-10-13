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

    def test_cloud_notebook_integrity_and_variable_dependencies(self):
        """Verify cloud notebook structure, cell order, and variable dependencies."""
        import json
        import ast

        nb_path = PROJECT_ROOT / "notebooks" / "01_rsna_yolo_cloud_training.ipynb"
        self.assertTrue(nb_path.exists(), f"Notebook not found at {nb_path}")

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        self.assertEqual(nb.get("nbformat"), 4)
        cells = nb.get("cells", [])
        self.assertGreater(len(cells), 10)

        # Ensure no pre-executed fake outputs are baked into the notebook
        for i, cell in enumerate(cells):
            if cell.get("cell_type") == "code":
                self.assertEqual(
                    cell.get("outputs", []),
                    [],
                    f"Code cell {i} must have empty outputs (no fabricated results)."
                )

        # Parse all code cells and inspect dependency sequence
        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        self.assertGreaterEqual(len(code_cells), 8)

        # 1. Environment Setup cell must define essential variables upfront
        env_code = "".join(code_cells[0].get("source", []))
        self.assertIn("compute_device", env_code)
        self.assertIn("YOLO_DATA_DIR", env_code)
        self.assertIn("yaml_path", env_code)
        self.assertIn("EXPERIMENT_DIR", env_code)
        self.assertIn("RUN_NAME", env_code)

        # 2. Smoke test cell must be self-contained
        smoke_code = "".join(code_cells[1].get("source", []))
        self.assertIn("smoke_test", smoke_code.lower())
        self.assertNotIn("YOLO_DATA_DIR", smoke_code, "Smoke test must not depend on YOLO_DATA_DIR.")
        self.assertNotIn("labels_csv_path", smoke_code, "Smoke test must not depend on labels_csv_path.")
        self.assertIn("smoke_yaml", smoke_code, "Smoke test must create its own YAML config.")

        # 3. Check training cell hyperparameters
        train_cell_code = "".join(code_cells[6].get("source", []))
        self.assertIn("epochs=25", train_cell_code, "Training must run for 25 epochs.")
        self.assertIn("save=True", train_cell_code)
        self.assertIn("save_period=1", train_cell_code)
        self.assertIn("resume=True", train_cell_code)

        # 4. AST syntax parse all code cells
        for idx, cell in enumerate(code_cells):
            src = "".join(cell.get("source", []))
            py_lines = [l for l in src.splitlines() if not l.strip().startswith("!") and not l.strip().startswith("%")]
            try:
                ast.parse("\n".join(py_lines))
            except SyntaxError as e:
                self.fail(f"Syntax error in code cell {idx}: {e}")


if __name__ == "__main__":
    unittest.main()


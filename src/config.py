"""Configuration module for the Medical Image Analysis project.

Defines global filepaths, constants, model hyper-parameters, and
hardware acceleration detection (Apple Silicon MPS / CUDA / CPU).
"""

from pathlib import Path
from typing import List
import os

# --- Project Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"

MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
DEFAULT_WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"
FALLBACK_PRETRAINED_MODEL = "yolov8s.pt"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREVIEWS_DIR = OUTPUTS_DIR / "ground_truth_previews"
REPORTS_DIR = PROJECT_ROOT / "reports"

# --- RSNA Dataset & Image Specifications ---
DICOM_IMAGE_SIZE = (1024, 1024)  # Native resolution of RSNA Chest X-Rays
YOLO_INPUT_SIZE = 640            # Standard input resolution for YOLO training/inference
CLASS_NAMES: List[str] = ["Lung Opacity"]
CLASS_ID_PNEUMONIA = 0

# --- Inference Defaults ---
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45


def get_inference_device() -> str:
    """Determine best available hardware accelerator for inference.

    Returns:
        str: 'mps' for Apple Silicon M1/M2/M3, 'cuda' for NVIDIA GPUs, or 'cpu'.
    """
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

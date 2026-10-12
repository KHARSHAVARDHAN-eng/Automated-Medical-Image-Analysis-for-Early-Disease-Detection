"""DICOM file utilities for reading and preprocessing medical radiographs.

Handles raw pixel extraction, photometric interpretation inversion (MONOCHROME1),
VOI LUT / windowing, and conversion to standard 8-bit images.
"""

from pathlib import Path
from typing import Dict, Any, Union, Tuple
import numpy as np


def read_dicom_image(
    dcm_path: Union[str, Path],
    target_size: Tuple[int, int] = None,
    to_rgb: bool = True
) -> np.ndarray:
    """Read a DICOM (.dcm) file and convert it into a normalized 8-bit image array.

    Args:
        dcm_path: Path to the .dcm radiograph file.
        target_size: Optional (width, height) tuple to resize image.
        to_rgb: If True, returns a 3-channel (H, W, 3) image; otherwise (H, W).

    Returns:
        np.ndarray: uint8 image array formatted for display or model input.
    """
    import pydicom
    try:
        from pydicom.pixels import apply_voi_lut
    except ImportError:
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    import cv2

    dcm = pydicom.dcmread(str(dcm_path))

    # Apply VOI LUT (Value of Interest Look-Up Table) if present
    try:
        pixel_array = apply_voi_lut(dcm.pixel_array, dcm)
    except Exception:
        pixel_array = dcm.pixel_array

    # Handle Photometric Interpretation (MONOCHROME1: low values are white)
    if hasattr(dcm, "PhotometricInterpretation") and dcm.PhotometricInterpretation == "MONOCHROME1":
        pixel_array = np.amax(pixel_array) - pixel_array

    # Min-max normalization to 0-255 uint8
    pixel_array = pixel_array.astype(float)
    min_val = np.min(pixel_array)
    max_val = np.max(pixel_array)

    if max_val > min_val:
        normalized = ((pixel_array - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    else:
        normalized = np.zeros(pixel_array.shape, dtype=np.uint8)

    # Optional resize
    if target_size is not None:
        normalized = cv2.resize(normalized, target_size, interpolation=cv2.INTER_AREA)

    # Format channels
    if to_rgb:
        if len(normalized.shape) == 2:
            return cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    return normalized


def get_dicom_metadata(dcm_path: Union[str, Path]) -> Dict[str, Any]:
    """Extract clinical and acquisition metadata from a DICOM header.

    Args:
        dcm_path: Path to the .dcm radiograph file.

    Returns:
        Dict[str, Any]: Dictionary containing PatientID, ViewPosition, Age, Sex, etc.
    """
    import pydicom

    dcm = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)

    return {
        "patient_id": getattr(dcm, "PatientID", "Unknown"),
        "patient_age": getattr(dcm, "PatientAge", "Unknown"),
        "patient_sex": getattr(dcm, "PatientSex", "Unknown"),
        "view_position": getattr(dcm, "ViewPosition", "Unknown"),  # AP or PA
        "modality": getattr(dcm, "Modality", "CR"),
        "rows": getattr(dcm, "Rows", None),
        "columns": getattr(dcm, "Columns", None),
    }

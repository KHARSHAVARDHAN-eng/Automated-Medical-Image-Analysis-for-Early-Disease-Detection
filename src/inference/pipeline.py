"""End-to-end inference pipeline for pneumonia opacity detection and reporting.

Combines DICOM ingestion, YOLO object detection inference, and clinical visualization
into a single callable interface.
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np

from src.data.dicom_utils import read_dicom_image, get_dicom_metadata
from src.models.detector import PneumoniaDetector
from src.visualization.visualizer import overlay_boxes, create_side_by_side_comparison
from src.config import DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD


class PneumoniaInferencePipeline:
    """Unified pipeline for processing chest X-rays and producing annotated outputs."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ) -> None:
        """Initialize pipeline with detector."""
        self.detector = PneumoniaDetector(weights_path=weights_path, device=device)

    def process(
        self,
        input_source: Union[str, Path, np.ndarray],
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> Dict[str, Any]:
        """Run complete diagnostic pipeline on an input image or DICOM file.

        Args:
            input_source: Path to .dcm/.png/.jpg or a numpy image array.
            conf_threshold: Detection confidence cutoff.
            iou_threshold: NMS overlap cutoff.

        Returns:
            Dict[str, Any]:
                - "has_pneumonia": bool
                - "detection_count": int
                - "max_confidence": float
                - "detections": list of bounding box dicts
                - "metadata": clinical DICOM metadata dict (if DICOM, else empty)
                - "original_image": RGB numpy array
                - "annotated_image": Annotated RGB numpy array
                - "side_by_side": Side-by-side comparison RGB numpy array
        """
        metadata: Dict[str, Any] = {}

        # 1. Load image
        if isinstance(input_source, (str, Path)):
            path = Path(input_source)
            if not path.exists():
                raise FileNotFoundError(f"Input file not found: {path}")

            if path.suffix.lower() in [".dcm", ".dicom"]:
                original_img = read_dicom_image(path)
                try:
                    metadata = get_dicom_metadata(path)
                except Exception:
                    metadata = {}
            else:
                import cv2
                bgr = cv2.imread(str(path))
                if bgr is None:
                    raise ValueError(f"Could not read image file: {path}")
                original_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(input_source, np.ndarray):
            original_img = input_source.copy()
        else:
            raise TypeError(f"Unsupported input source type: {type(input_source)}")

        # 2. Run detector
        detection_result = self.detector.predict(
            image=original_img,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

        # 3. Create visualizations
        annotated_img = overlay_boxes(original_img, detection_result["detections"])
        side_by_side = create_side_by_side_comparison(original_img, annotated_img)

        return {
            "has_pneumonia": detection_result["has_pneumonia"],
            "detection_count": detection_result["detection_count"],
            "max_confidence": detection_result["max_confidence"],
            "detections": detection_result["detections"],
            "device_used": detection_result["device_used"],
            "metadata": metadata,
            "original_image": original_img,
            "annotated_image": annotated_img,
            "side_by_side": side_by_side,
        }

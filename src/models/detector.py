"""Pneumonia object detection model wrapper based on Ultralytics YOLO.

Manages weight loading, device placement (MPS / CUDA / CPU), and structured
prediction outputs for chest radiographs.
"""

from pathlib import Path
from typing import Dict, List, Any, Union, Optional
import numpy as np

from src.config import (
    DEFAULT_WEIGHTS_PATH,
    FALLBACK_PRETRAINED_MODEL,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    CLASS_NAMES,
    get_inference_device,
)


class PneumoniaDetector:
    """Wrapper around YOLOv8 for detecting pneumonia-induced lung opacities."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ) -> None:
        """Initialize detector with model weights and device.

        Args:
            weights_path: Path to trained .pt weights. If None or not found,
                          defaults to DEFAULT_WEIGHTS_PATH or FALLBACK_PRETRAINED_MODEL.
            device: 'mps', 'cuda', or 'cpu'. If None, auto-detected.
        """
        self.device = device or get_inference_device()
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
        self._model = None

    @property
    def model(self):
        """Lazy loader for YOLO model instance."""
        if self._model is None:
            from ultralytics import YOLO

            target_weights = self.weights_path
            if not target_weights.exists():
                # Provide guidance if weights are not yet present locally
                print(
                    f"[PneumoniaDetector] Local weights '{target_weights}' not found. "
                    f"Falling back to pretrained architecture '{FALLBACK_PRETRAINED_MODEL}' for scaffolding/testing."
                )
                target_weights = FALLBACK_PRETRAINED_MODEL

            self._model = YOLO(str(target_weights))
        return self._model

    def predict(
        self,
        image: Union[np.ndarray, str, Path],
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> Dict[str, Any]:
        """Run object detection on an input image.

        Args:
            image: Image array (H, W, 3) or filepath string/Path.
            conf_threshold: Minimum detection confidence score.
            iou_threshold: Intersection-over-Union threshold for NMS.

        Returns:
            Dict[str, Any]:
                - "has_pneumonia": bool
                - "detection_count": int
                - "detections": list of dicts with box coords, conf, class_name
                - "max_confidence": float
        """
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        max_conf = 0.0

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    class_name = CLASS_NAMES[0] if cls_id == 0 else f"Class_{cls_id}"
                    if conf > max_conf:
                        max_conf = conf

                    detections.append({
                        "box_xyxy": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                        "confidence": round(conf, 4),
                        "class_id": cls_id,
                        "class_name": class_name,
                    })

        has_pneumonia = len(detections) > 0

        return {
            "has_pneumonia": has_pneumonia,
            "detection_count": len(detections),
            "max_confidence": round(max_conf, 4),
            "detections": detections,
            "device_used": self.device,
        }

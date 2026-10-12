"""Medical radiograph visualization tools.

Provides high-contrast bounding box overlays, diagnostic badge annotations,
and side-by-side comparative views for clinical review.
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


def overlay_boxes(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    box_color: Tuple[int, int, int] = (255, 60, 60),  # High-contrast red/coral
    text_color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2
) -> np.ndarray:
    """Draw bounding boxes and confidence score labels onto a copy of the image.

    Args:
        image: Original radiograph image (H, W, 3) in RGB.
        detections: List of detection dicts with 'box_xyxy', 'confidence', 'class_name'.
        box_color: RGB tuple for the bounding box.
        text_color: RGB tuple for the text inside badge.
        thickness: Box outline pixel thickness.

    Returns:
        np.ndarray: Annotated RGB image copy.
    """
    annotated = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det["box_xyxy"]
        conf = det.get("confidence", 0.0)
        label = det.get("class_name", "Lung Opacity")
        caption = f"{label} {conf:.1%}" if conf > 0 else label

        # Draw main bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

        # Compute label text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (tw, th), baseline = cv2.getTextSize(caption, font, font_scale, font_thickness)

        # Draw background badge for text readability
        badge_y1 = max(0, y1 - th - baseline - 4)
        badge_y2 = y1
        badge_x2 = min(annotated.shape[1], x1 + tw + 6)
        cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), box_color, -1)

        # Draw caption text inside badge
        text_origin = (x1 + 3, y1 - baseline - 2)
        cv2.putText(
            annotated,
            caption,
            text_origin,
            font,
            font_scale,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return annotated


def create_side_by_side_comparison(
    original_img: np.ndarray,
    annotated_img: np.ndarray,
    divider_width: int = 4
) -> np.ndarray:
    """Combine original and annotated images horizontally with a separation divider.

    Args:
        original_img: Original radiograph (H, W, 3).
        annotated_img: Annotated radiograph (H, W, 3).
        divider_width: Pixel width of dividing vertical line.

    Returns:
        np.ndarray: Combined side-by-side image.
    """
    h1, w1 = original_img.shape[:2]
    h2, w2 = annotated_img.shape[:2]

    # Ensure heights match
    if h1 != h2:
        annotated_img = cv2.resize(annotated_img, (w2, h1))

    divider = np.zeros((h1, divider_width, 3), dtype=np.uint8)
    divider[:] = (80, 80, 80)  # Neutral gray separator

    return np.hstack([original_img, divider, annotated_img])

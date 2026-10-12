"""Unit tests for visualization utilities."""

import unittest
import numpy as np
from src.visualization.visualizer import overlay_boxes, create_side_by_side_comparison


class TestVisualizer(unittest.TestCase):
    """Test suite for visualization routines."""

    def test_overlay_boxes(self):
        """Verify that bounding box overlay generates a valid RGB array of equal shape."""
        dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
        detections = [
            {
                "box_xyxy": [100, 100, 250, 300],
                "confidence": 0.88,
                "class_name": "Lung Opacity",
            }
        ]

        annotated = overlay_boxes(dummy_img, detections)

        self.assertEqual(annotated.shape, dummy_img.shape)
        self.assertEqual(annotated.dtype, np.uint8)
        self.assertFalse(np.array_equal(dummy_img, annotated))

    def test_create_side_by_side_comparison(self):
        """Verify horizontal concatenation with separator."""
        img1 = np.zeros((400, 400, 3), dtype=np.uint8)
        img2 = np.ones((400, 400, 3), dtype=np.uint8) * 255
        divider_w = 4

        side_by_side = create_side_by_side_comparison(img1, img2, divider_width=divider_w)

        expected_width = 400 + divider_w + 400
        self.assertEqual(side_by_side.shape, (400, expected_width, 3))


if __name__ == "__main__":
    unittest.main()

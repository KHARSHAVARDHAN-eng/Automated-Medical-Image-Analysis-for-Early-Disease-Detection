"""Unit tests for RSNA to YOLO coordinate conversions."""

import unittest
from src.data.rsna_parser import convert_bbox_to_yolo, convert_yolo_to_bbox


class TestCoordinates(unittest.TestCase):
    """Test suite for coordinate math transformations."""

    def test_bbox_to_yolo_conversion(self):
        """Test converting standard RSNA bounding box to normalized YOLO format."""
        # Image 1024x1024, box at (200, 300) with width 100, height 200
        cls_id, xc, yc, w, h = convert_bbox_to_yolo(
            x=200, y=300, width=100, height=200, img_width=1024, img_height=1024, class_id=0
        )

        self.assertEqual(cls_id, 0)
        # Center x: (200 + 50) / 1024 = 250 / 1024
        self.assertAlmostEqual(xc, 250.0 / 1024.0, places=4)
        # Center y: (300 + 100) / 1024 = 400 / 1024
        self.assertAlmostEqual(yc, 400.0 / 1024.0, places=4)
        self.assertAlmostEqual(w, 100.0 / 1024.0, places=4)
        self.assertAlmostEqual(h, 200.0 / 1024.0, places=4)

    def test_round_trip_conversion(self):
        """Verify converting to YOLO and back to pixel coordinates preserves bounding box."""
        orig_x, orig_y, orig_w, orig_h = 150, 220, 180, 260
        cls_id, xc, yc, nw, nh = convert_bbox_to_yolo(
            orig_x, orig_y, orig_w, orig_h, img_width=1024, img_height=1024
        )

        x1, y1, x2, y2 = convert_yolo_to_bbox(
            xc, yc, nw, nh, img_width=1024, img_height=1024
        )

        self.assertLessEqual(abs(x1 - orig_x), 1)
        self.assertLessEqual(abs(y1 - orig_y), 1)
        self.assertLessEqual(abs((x2 - x1) - orig_w), 1)
        self.assertLessEqual(abs((y2 - y1) - orig_h), 1)


if __name__ == "__main__":
    unittest.main()

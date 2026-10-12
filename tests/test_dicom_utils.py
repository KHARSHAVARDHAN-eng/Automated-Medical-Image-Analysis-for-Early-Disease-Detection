"""Unit tests for DICOM decoding, VOI LUT, MONOCHROME inversion, and metadata extraction."""

import unittest
import tempfile
from pathlib import Path
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

from src.data.dicom_utils import read_dicom_image, get_dicom_metadata


class TestDicomUtils(unittest.TestCase):
    """Test suite for DICOM handling routines."""

    def setUp(self):
        """Create a temporary synthetic DICOM file for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dcm_path = Path(self.temp_dir.name) / "synthetic_test.dcm"

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(str(self.dcm_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.PatientID = "TEST_PATIENT_999"
        ds.PatientAge = "052Y"
        ds.PatientSex = "F"
        ds.ViewPosition = "AP"
        ds.Modality = "CR"
        ds.PhotometricInterpretation = "MONOCHROME1"
        ds.Rows = 1024
        ds.Columns = 1024
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1

        # Pixel ramp: corners are 0 and 4000
        pixel_array = np.zeros((1024, 1024), dtype=np.uint16)
        pixel_array[0, 0] = 0
        pixel_array[1023, 1023] = 4000
        ds.PixelData = pixel_array.tobytes()
        ds.save_as(str(self.dcm_path))

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_read_dicom_shape_and_dtype(self):
        """Verify decoded DICOM returns (1024, 1024, 3) uint8 image."""
        img = read_dicom_image(self.dcm_path)
        self.assertEqual(img.shape, (1024, 1024, 3))
        self.assertEqual(img.dtype, np.uint8)

    def test_monochrome1_inversion(self):
        """Verify MONOCHROME1 inverts pixel values (min becomes max)."""
        img = read_dicom_image(self.dcm_path)
        # Original (0, 0) was min (0). Under MONOCHROME1 inversion, min becomes white (255)
        # Original (1023, 1023) was max (4000). Under MONOCHROME1 inversion, max becomes black (0)
        self.assertEqual(img[0, 0, 0], 255)
        self.assertEqual(img[1023, 1023, 0], 0)

    def test_target_size_resize(self):
        """Verify optional resizing to YOLO dimensions."""
        img = read_dicom_image(self.dcm_path, target_size=(640, 640))
        self.assertEqual(img.shape, (640, 640, 3))

    def test_get_dicom_metadata(self):
        """Verify extraction of clinical header attributes."""
        meta = get_dicom_metadata(self.dcm_path)
        self.assertEqual(meta["patient_id"], "TEST_PATIENT_999")
        self.assertEqual(meta["patient_age"], "052Y")
        self.assertEqual(meta["patient_sex"], "F")
        self.assertEqual(meta["view_position"], "AP")
        self.assertEqual(meta["rows"], 1024)
        self.assertEqual(meta["columns"], 1024)

    def test_missing_file_handling(self):
        """Verify handling when DICOM file does not exist."""
        non_existent = Path(self.temp_dir.name) / "does_not_exist.dcm"
        with self.assertRaises(Exception):
            read_dicom_image(non_existent)


if __name__ == "__main__":
    unittest.main()

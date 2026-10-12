# RSNA Pneumonia Detection: Dataset Verification Report

- **Verification Timestamp:** 2026-10-12 08:19:44
- **Dataset:** RSNA Pneumonia Detection Challenge (`stage_2_train_labels.csv`)
- **Total Rows in Official CSV:** 30,227
- **Platform:** Apple MacBook M1 (8 GB Unified Memory) | Antigravity IDE

---

## 1. Verified Real RSNA Samples

| Patient ID | DICOM File | Status Label | Target | Boxes | Dimensions | Modality / View | Preview Image |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `0004cfab-14fd-4e49-80ba-63a80b6bddd6` | `0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm` | **NEGATIVE (Normal / No Opacity)** | `0` | `0` | 1024 x 1024 | CR / PA | [`preview_0004cfab-14fd-4e49-80ba-63a80b6bddd6.png`](outputs/ground_truth_previews/preview_0004cfab-14fd-4e49-80ba-63a80b6bddd6.png) |
| `00436515-870c-4b36-a041-de91049b9ab4` | `00436515-870c-4b36-a041-de91049b9ab4.dcm` | **POSITIVE (Pneumonia Detected)** | `1` | `2` | 1024 x 1024 | CR / AP | [`preview_00436515-870c-4b36-a041-de91049b9ab4.png`](outputs/ground_truth_previews/preview_00436515-870c-4b36-a041-de91049b9ab4.png) |

---

## 2. Detailed Ground-Truth Bounding Box Coordinates

### Patient: `0004cfab-14fd-4e49-80ba-63a80b6bddd6` (NEGATIVE (Normal / No Opacity))
- **Patient Demographics:** Age 51, Sex F
- **Acquisition View:** PA (CR)
- **Native Resolution:** 1024 x 1024
- **Total Annotated Opacity Boxes:** 0
- **Annotations:** None (Confirmed Normal Chest Radiograph).

### Patient: `00436515-870c-4b36-a041-de91049b9ab4` (POSITIVE (Pneumonia Detected))
- **Patient Demographics:** Age 32, Sex F
- **Acquisition View:** AP (CR)
- **Native Resolution:** 1024 x 1024
- **Total Annotated Opacity Boxes:** 2

| Box # | Top-Left X (`px`) | Top-Left Y (`px`) | Width (`px`) | Height (`px`) | Normalized YOLO (xc, yc, w, h) |
| :---: | :---: | :---: | :---: | :---: | :--- |
| #1 | `264.0` | `152.0` | `213.0` | `379.0` | `0 0.3618 0.3335 0.2080 0.3701` |
| #2 | `562.0` | `152.0` | `256.0` | `453.0` | `0 0.6738 0.3696 0.2500 0.4424` |

---

## 3. Preprocessing & Quality Checks

- [x] **DICOM Pixel Decoding:** Successfully decoded 16-bit radiograph pixel arrays using `pydicom`.
- [x] **Windowing / VOI LUT:** Successfully mapped high-dynamic-range radiograph intensities to 8-bit visual range.
- [x] **Photometric Interpretation:** Inversion applied when required (`MONOCHROME1` to `MONOCHROME2`) ensuring bone and opacities render white.
- [x] **Coordinate Transformation:** Verified bidirectional math between absolute pixel bounding boxes and normalized YOLO format.
- [x] **Ground Truth vs. Prediction Distinction:** Ground truth rendered with emerald green badges (`Ground Truth: Lung Opacity #N`), keeping predictions strictly separated.
- [x] **Zero Model Training Claim:** Model inference has not been run or fabricated. Results represent genuine clinical radiologist ground truth.

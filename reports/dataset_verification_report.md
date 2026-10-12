# RSNA Pneumonia Detection: Dataset Verification Report

- **Verification Timestamp:** 2026-10-12 09:49:05
- **Dataset:** RSNA Pneumonia Detection Challenge (`stage_2_train_labels.csv`)
- **Total Rows in Official CSV:** 30,227
- **Platform:** Apple MacBook M1 (8 GB Unified Memory) | Antigravity IDE

---

## 1. Verified Real RSNA Samples

| Patient ID | DICOM File | Status Label | Target | Boxes | Dimensions | Modality / View | Preview Image |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `0004cfab-14fd-4e49-80ba-63a80b6bddd6` | `0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm` | **NEGATIVE (Normal / No Opacity)** | `0` | `0` | 1024 x 1024 | CR / PA | [`preview_0004cfab-14fd-4e49-80ba-63a80b6bddd6.png`](outputs/ground_truth_previews/preview_0004cfab-14fd-4e49-80ba-63a80b6bddd6.png) |
| `00313ee0-9eaa-42f4-b0ab-c148ed3241cd` | `00313ee0-9eaa-42f4-b0ab-c148ed3241cd.dcm` | **NEGATIVE (Normal / No Opacity)** | `0` | `0` | 1024 x 1024 | CR / PA | [`preview_00313ee0-9eaa-42f4-b0ab-c148ed3241cd.png`](outputs/ground_truth_previews/preview_00313ee0-9eaa-42f4-b0ab-c148ed3241cd.png) |
| `00322d4d-1c29-4943-afc9-b6754be640eb` | `00322d4d-1c29-4943-afc9-b6754be640eb.dcm` | **NEGATIVE (Normal / No Opacity)** | `0` | `0` | 1024 x 1024 | CR / AP | [`preview_00322d4d-1c29-4943-afc9-b6754be640eb.png`](outputs/ground_truth_previews/preview_00322d4d-1c29-4943-afc9-b6754be640eb.png) |
| `003d8fa0-6bf1-40ed-b54c-ac657f8495c5` | `003d8fa0-6bf1-40ed-b54c-ac657f8495c5.dcm` | **NEGATIVE (Normal / No Opacity)** | `0` | `0` | 1024 x 1024 | CR / PA | [`preview_003d8fa0-6bf1-40ed-b54c-ac657f8495c5.png`](outputs/ground_truth_previews/preview_003d8fa0-6bf1-40ed-b54c-ac657f8495c5.png) |
| `00436515-870c-4b36-a041-de91049b9ab4` | `00436515-870c-4b36-a041-de91049b9ab4.dcm` | **POSITIVE (Pneumonia Detected)** | `1` | `2` | 1024 x 1024 | CR / AP | [`preview_00436515-870c-4b36-a041-de91049b9ab4.png`](outputs/ground_truth_previews/preview_00436515-870c-4b36-a041-de91049b9ab4.png) |
| `00569f44-917d-4c86-a842-81832af98c30` | `00569f44-917d-4c86-a842-81832af98c30.dcm` | **NEGATIVE (Normal / No Opacity)** | `0` | `0` | 1024 x 1024 | CR / AP | [`preview_00569f44-917d-4c86-a842-81832af98c30.png`](outputs/ground_truth_previews/preview_00569f44-917d-4c86-a842-81832af98c30.png) |
| `00704310-78a8-4b38-8475-49f4573b2dbb` | `00704310-78a8-4b38-8475-49f4573b2dbb.dcm` | **POSITIVE (Pneumonia Detected)** | `1` | `2` | 1024 x 1024 | CR / PA | [`preview_00704310-78a8-4b38-8475-49f4573b2dbb.png`](outputs/ground_truth_previews/preview_00704310-78a8-4b38-8475-49f4573b2dbb.png) |
| `00aecb01-a116-45a2-956c-08d2fa55433f` | `00aecb01-a116-45a2-956c-08d2fa55433f.dcm` | **POSITIVE (Pneumonia Detected)** | `1` | `2` | 1024 x 1024 | CR / AP | [`preview_00aecb01-a116-45a2-956c-08d2fa55433f.png`](outputs/ground_truth_previews/preview_00aecb01-a116-45a2-956c-08d2fa55433f.png) |
| `00c0b293-48e7-4e16-ac76-9269ba535a62` | `00c0b293-48e7-4e16-ac76-9269ba535a62.dcm` | **POSITIVE (Pneumonia Detected)** | `1` | `2` | 1024 x 1024 | CR / AP | [`preview_00c0b293-48e7-4e16-ac76-9269ba535a62.png`](outputs/ground_truth_previews/preview_00c0b293-48e7-4e16-ac76-9269ba535a62.png) |
| `00f08de1-517e-4652-a04f-d1dc9ee48593` | `00f08de1-517e-4652-a04f-d1dc9ee48593.dcm` | **POSITIVE (Pneumonia Detected)** | `1` | `2` | 1024 x 1024 | CR / AP | [`preview_00f08de1-517e-4652-a04f-d1dc9ee48593.png`](outputs/ground_truth_previews/preview_00f08de1-517e-4652-a04f-d1dc9ee48593.png) |

---

## 2. Detailed Ground-Truth Bounding Box Coordinates

### Patient: `0004cfab-14fd-4e49-80ba-63a80b6bddd6` (NEGATIVE (Normal / No Opacity))
- **Patient Demographics:** Age 51, Sex F
- **Acquisition View:** PA (CR)
- **Native Resolution:** 1024 x 1024
- **Total Annotated Opacity Boxes:** 0
- **Annotations:** None (Confirmed Normal Chest Radiograph).

### Patient: `00313ee0-9eaa-42f4-b0ab-c148ed3241cd` (NEGATIVE (Normal / No Opacity))
- **Patient Demographics:** Age 48, Sex F
- **Acquisition View:** PA (CR)
- **Native Resolution:** 1024 x 1024
- **Total Annotated Opacity Boxes:** 0
- **Annotations:** None (Confirmed Normal Chest Radiograph).

### Patient: `00322d4d-1c29-4943-afc9-b6754be640eb` (NEGATIVE (Normal / No Opacity))
- **Patient Demographics:** Age 19, Sex M
- **Acquisition View:** AP (CR)
- **Native Resolution:** 1024 x 1024
- **Total Annotated Opacity Boxes:** 0
- **Annotations:** None (Confirmed Normal Chest Radiograph).

### Patient: `003d8fa0-6bf1-40ed-b54c-ac657f8495c5` (NEGATIVE (Normal / No Opacity))
- **Patient Demographics:** Age 28, Sex M
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

### Patient: `00569f44-917d-4c86-a842-81832af98c30` (NEGATIVE (Normal / No Opacity))
- **Patient Demographics:** Age 54, Sex M
- **Acquisition View:** AP (CR)
- **Native Resolution:** 1024 x 1024
- **Total Annotated Opacity Boxes:** 0
- **Annotations:** None (Confirmed Normal Chest Radiograph).

### Patient: `00704310-78a8-4b38-8475-49f4573b2dbb` (POSITIVE (Pneumonia Detected))
- **Patient Demographics:** Age 75, Sex M
- **Acquisition View:** PA (CR)
- **Native Resolution:** 1024 x 1024
- **Total Annotated Opacity Boxes:** 2

| Box # | Top-Left X (`px`) | Top-Left Y (`px`) | Width (`px`) | Height (`px`) | Normalized YOLO (xc, yc, w, h) |
| :---: | :---: | :---: | :---: | :---: | :--- |
| #1 | `323.0` | `577.0` | `160.0` | `104.0` | `0 0.3936 0.6143 0.1562 0.1016` |
| #2 | `695.0` | `575.0` | `162.0` | `137.0` | `0 0.7578 0.6284 0.1582 0.1338` |

### Patient: `00aecb01-a116-45a2-956c-08d2fa55433f` (POSITIVE (Pneumonia Detected))
- **Patient Demographics:** Age 6, Sex F
- **Acquisition View:** AP (CR)
- **Native Resolution:** 1024 x 1024
- **Total Annotated Opacity Boxes:** 2

| Box # | Top-Left X (`px`) | Top-Left Y (`px`) | Width (`px`) | Height (`px`) | Normalized YOLO (xc, yc, w, h) |
| :---: | :---: | :---: | :---: | :---: | :--- |
| #1 | `288.0` | `322.0` | `94.0` | `135.0` | `0 0.3271 0.3804 0.0918 0.1318` |
| #2 | `547.0` | `299.0` | `119.0` | `165.0` | `0 0.5923 0.3726 0.1162 0.1611` |

### Patient: `00c0b293-48e7-4e16-ac76-9269ba535a62` (POSITIVE (Pneumonia Detected))
- **Patient Demographics:** Age 23, Sex F
- **Acquisition View:** AP (CR)
- **Native Resolution:** 1024 x 1024
- **Total Annotated Opacity Boxes:** 2

| Box # | Top-Left X (`px`) | Top-Left Y (`px`) | Width (`px`) | Height (`px`) | Normalized YOLO (xc, yc, w, h) |
| :---: | :---: | :---: | :---: | :---: | :--- |
| #1 | `306.0` | `544.0` | `168.0` | `244.0` | `0 0.3809 0.6504 0.1641 0.2383` |
| #2 | `650.0` | `511.0` | `206.0` | `284.0` | `0 0.7354 0.6377 0.2012 0.2773` |

### Patient: `00f08de1-517e-4652-a04f-d1dc9ee48593` (POSITIVE (Pneumonia Detected))
- **Patient Demographics:** Age 58, Sex M
- **Acquisition View:** AP (CR)
- **Native Resolution:** 1024 x 1024
- **Total Annotated Opacity Boxes:** 2

| Box # | Top-Left X (`px`) | Top-Left Y (`px`) | Width (`px`) | Height (`px`) | Normalized YOLO (xc, yc, w, h) |
| :---: | :---: | :---: | :---: | :---: | :--- |
| #1 | `181.0` | `184.0` | `206.0` | `506.0` | `0 0.2773 0.4268 0.2012 0.4941` |
| #2 | `571.0` | `275.0` | `230.0` | `476.0` | `0 0.6699 0.5010 0.2246 0.4648` |

---

## 3. Preprocessing & Quality Checks

- [x] **DICOM Pixel Decoding:** Successfully decoded 16-bit radiograph pixel arrays using `pydicom`.
- [x] **Windowing / VOI LUT:** Successfully mapped high-dynamic-range radiograph intensities to 8-bit visual range.
- [x] **Photometric Interpretation:** Inversion applied when required (`MONOCHROME1` to `MONOCHROME2`) ensuring bone and opacities render white.
- [x] **Coordinate Transformation:** Verified bidirectional math between absolute pixel bounding boxes and normalized YOLO format.
- [x] **Ground Truth vs. Prediction Distinction:** Ground truth rendered with emerald green badges (`Ground Truth: Lung Opacity #N`), keeping predictions strictly separated.
- [x] **Zero Model Training Claim:** Model inference has not been run or fabricated. Results represent genuine clinical radiologist ground truth.

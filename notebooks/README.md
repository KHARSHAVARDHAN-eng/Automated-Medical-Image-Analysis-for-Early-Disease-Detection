# Cloud Training Notebooks (Kaggle / Google Colab)

This directory contains cloud-optimized Jupyter notebooks designed to execute GPU-intensive tasks on platforms like **Kaggle Notebooks** (recommended) or **Google Colab** (with free T4 GPU).

## Notebooks Overview

- [`01_rsna_yolo_cloud_training.ipynb`](file:///Users/harsha/Desktop/Disease_Detection_Iva/notebooks/01_rsna_yolo_cloud_training.ipynb):
  - Sets up the environment with GPU acceleration (CUDA).
  - Accesses the RSNA Pneumonia Detection Challenge dataset directly from Kaggle input directory.
  - Converts DICOM X-rays to standard 8-bit images (640x640 or 1024x1024).
  - Converts bounding boxes `[x, y, w, h]` to normalized YOLO format `[class, x_center, y_center, w, h]`.
  - Generates `rsna_yolo.yaml` configuration.
  - Trains YOLOv8 (e.g. `yolov8s.pt` or `yolov8m.pt`) with transfer learning.
  - Evaluates performance using Mean Average Precision (`mAP50`, `mAP50-95`).
  - Exports the best weights (`best.pt`) for local inference on your Apple M1 Mac.

---

## How to Run on Kaggle (Recommended)

1. Navigate to [kaggle.com/code](https://www.kaggle.com/code) and click **"New Notebook"**.
2. Under **Notebook Options** (right sidebar):
   - **Accelerator**: Select **GPU T4 x2** or **GPU P100**.
   - **Internet**: Toggle **On**.
3. Under **Input Data**:
   - Click **"Add Data"** -> search for **"RSNA Pneumonia Detection Challenge"** (`kaggle competitions download -c rsna-pneumonia-detection-challenge`).
   - The dataset will be instantly mounted under `/kaggle/input/rsna-pneumonia-detection-challenge` without manual downloading.
4. Upload `01_rsna_yolo_cloud_training.ipynb` via **File -> Upload Notebook**.
5. Execute cells sequentially.
6. Once training completes, download `best.pt` from `/kaggle/working/runs/detect/train/weights/best.pt` and place it in your local `models/weights/best.pt`.

---

## How to Run on Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Choose **Upload** and select `01_rsna_yolo_cloud_training.ipynb`.
3. Set runtime to GPU: **Runtime -> Change runtime type -> T4 GPU**.
4. Upload your `kaggle.json` API token to download the RSNA competition dataset.
5. Run the cells and save `best.pt` to Google Drive or download directly.

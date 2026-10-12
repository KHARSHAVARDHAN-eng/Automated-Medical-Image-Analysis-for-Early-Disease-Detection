# Cloud GPU Training Notebooks (Kaggle & Google Colab)

This directory contains cloud-optimized Jupyter notebooks designed to execute GPU-intensive deep learning tasks on **Kaggle Notebooks** (primary) or **Google Colab** (secondary).

---

## 🚀 Notebook Overview

### [`01_rsna_yolo_cloud_training.ipynb`](file:///Users/harsha/Desktop/Disease_Detection_Iva/notebooks/01_rsna_yolo_cloud_training.ipynb)
- **Class Configuration**: Single-class detection: `0: pneumonia_opacity`.
- **Hardware Acceleration**: Automatic CUDA detection with multi-core DICOM preprocessing.
- **Patient-Level Stratification**: 80% Train, 10% Validation, 10% Test partitions with **zero patient data leakage**.
- **Dataset Generation**: Converts native 16-bit DICOM radiographs into 640x640 JPEGs. Creates normalized `.txt` labels for positive cases and empty `.txt` files for negative normal cases (negative background learning).
- **Integrity Validation**: Runs automated 1:1 image-label verification and split disjointness checks prior to model initialization.
- **Model Architecture**: Ultralytics YOLOv8 (`yolov8s.pt` or `yolov8n.pt`).
- **Persistent Checkpointing**:
  - Saves weights after each epoch (`save=True`, `save_period=1`).
  - Preserves best weights (`best.pt`) and last weights (`last.pt`).
  - Supports resuming interrupted runs seamlessly via `YOLO("last.pt").train(resume=True)`.
- **Metrics & Diagnostic Outputs**:
  - Object detection: Precision, Recall, `mAP@0.50`, `mAP@0.50:0.95`.
  - Image-level classification: Accuracy, Sensitivity (Recall), Specificity, and Confusion Matrix.
  - Generates training loss curves (`results.png`), confusion matrix (`confusion_matrix.png`), and prediction sample overlays.

---

## 🛠️ Step-by-Step Execution on Kaggle (Primary)

1. Open [Kaggle Code](https://www.kaggle.com/code) and click **"New Notebook"**.
2. **Configure Environment** (Right Sidebar):
   - **Accelerator**: Select **GPU T4 x2** or **GPU P100**.
   - **Internet**: Toggle **On** (required for `pip install ultralytics`).
3. **Attach RSNA Competition Dataset**:
   - Under **Input** in the right sidebar, click **"Add Input"**.
   - Search for **"RSNA Pneumonia Detection Challenge"** (or click Competitions -> RSNA Pneumonia Detection Challenge).
   - *Because you have accepted the competition rules, the dataset will instantly mount under `/kaggle/input/rsna-pneumonia-detection-challenge` without any manual downloading.*
4. **Upload Notebook**:
   - In the Kaggle notebook menu, go to **File -> Upload Notebook**.
   - Select [`notebooks/01_rsna_yolo_cloud_training.ipynb`](file:///Users/harsha/Desktop/Disease_Detection_Iva/notebooks/01_rsna_yolo_cloud_training.ipynb) from your local project folder.
5. **Execute Training**:
   - Click **Run All** (or execute cells sequentially).
   - Training runs at high speed on the cloud GPU (~30–45 minutes for 25 epochs).
6. **Download Trained Checkpoint (`best.pt`)**:
   - Go to the **Output** tab in the right sidebar.
   - Navigate to `/kaggle/working/rsna_pneumonia/yolov8s_pneumonia_run/weights/best.pt`.
   - Click **Download** and place `best.pt` in your local project directory at:
     ```text
     Disease_Detection_Iva/models/weights/best.pt
     ```

---

## 🔄 Backup Execution on Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Choose **Upload** and upload `01_rsna_yolo_cloud_training.ipynb`.
3. Set hardware accelerator: **Runtime -> Change runtime type -> T4 GPU**.
4. Upload your `kaggle.json` API key to download the dataset into the runtime.
5. Run all cells and download `best.pt` to your local `models/weights/` directory.

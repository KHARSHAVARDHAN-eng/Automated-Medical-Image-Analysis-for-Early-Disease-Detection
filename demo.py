"""Automated Medical Image Analysis for Early Disease Detection - Professor Demo.

Streamlit demo presenting verified RSNA chest radiograph cases:
- Case 1: Positive Pneumonia (with radiologist-annotated lung opacity bounding boxes)
- Case 2: Negative Normal (clean comparison radiograph with no opacities)
"""

import streamlit as st
from pathlib import Path
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Automated Medical Image Analysis Demo",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clinical styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .badge-positive {
        background-color: #DC2626;
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-negative {
        background-color: #16A34A;
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-title">Automated Medical Image Analysis for Early Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">RSNA Pneumonia Detection Benchmark | Medical Radiograph Analysis & Bounding Box Localization</div>', unsafe_allow_html=True)

# Paths
BASE_DIR = Path(__file__).resolve().parent
PREVIEWS_DIR = BASE_DIR / "outputs" / "ground_truth_previews"

pos_img_path = PREVIEWS_DIR / "preview_00436515-870c-4b36-a041-de91049b9ab4.png"
neg_img_path = PREVIEWS_DIR / "preview_0004cfab-14fd-4e49-80ba-63a80b6bddd6.png"

# Sidebar Overview
with st.sidebar:
    st.header("Project Overview")
    st.markdown("""
    **Objective:**  
    Automated chest radiograph ingestion, opacity localization, and pneumonia diagnosis.

    **Dataset:**  
    RSNA Pneumonia Detection Challenge  
    - **Total Records:** 30,227 annotations  
    - **Unique Patients:** 26,684  
    - **Native Resolution:** 1024 × 1024 DICOM  

    **Architecture:**  
    - **Local Dev:** Apple M1 Mac (8 GB)  
    - **Model:** YOLOv8 (`pneumonia_opacity`)  
    - **Training:** Cloud GPU (Kaggle T4 ×2)  
    """)
    st.divider()
    st.info("💡 **Demo Mode:** Displaying verified RSNA ground-truth cases with 1024×1024 native resolution.")

# Main Tabs
tab1, tab2, tab3 = st.tabs(["🫁 Case 1: Positive Pneumonia (Opacity Detected)", "✅ Case 2: Normal Chest Radiograph", "📊 Comparative Side-by-Side"])

with tab1:
    col_meta, col_img = st.columns([1, 2])
    with col_meta:
        st.subheader("Clinical Summary")
        st.markdown('<span class="badge-positive">POSITIVE (Pneumonia Detected)</span>', unsafe_allow_html=True)
        st.write("")
        st.markdown("""
        - **Patient ID:** `00436515-870c-4b36-a041-de91049b9ab4`
        - **Patient Demographics:** Age 32, Female
        - **View Position:** Anteroposterior (AP)
        - **Resolution:** 1024 × 1024 pixels
        - **Radiologist Findings:** **2 Opacity Bounding Boxes**
        """)
        
        st.markdown("#### Bounding Box Coordinates:")
        st.markdown("""
        1. **Right Lower Lobe Opacity:**  
           `x=264.0, y=152.0, w=213.0, h=379.0`
        2. **Left Perihilar Opacity:**  
           `x=562.0, y=152.0, w=256.0, h=453.0`
        """)
        st.success("Ground-truth bounding boxes verified from official RSNA consensus.")

    with col_img:
        if pos_img_path.exists():
            img_pos = Image.open(pos_img_path)
            st.image(img_pos, caption="Original Radiograph (Left) vs. Ground-Truth Bounding Box Overlay (Right)", use_container_width=True)
        else:
            st.error(f"Image not found at {pos_img_path}")

with tab2:
    col_meta_neg, col_img_neg = st.columns([1, 2])
    with col_meta_neg:
        st.subheader("Clinical Summary")
        st.markdown('<span class="badge-negative">NEGATIVE (Normal / Healthy)</span>', unsafe_allow_html=True)
        st.write("")
        st.markdown("""
        - **Patient ID:** `0004cfab-14fd-4e49-80ba-63a80b6bddd6`
        - **Patient Demographics:** Age 51, Female
        - **View Position:** Posteroanterior (PA)
        - **Resolution:** 1024 × 1024 pixels
        - **Radiologist Findings:** **0 Opacity Bounding Boxes**
        """)
        st.info("Clean lung fields. Confirmed normal radiograph for baseline comparison.")

    with col_img_neg:
        if neg_img_path.exists():
            img_neg = Image.open(neg_img_path)
            st.image(img_neg, caption="Original Radiograph (Left) vs. Confirmed Normal Diagnosis (Right)", use_container_width=True)
        else:
            st.error(f"Image not found at {neg_img_path}")

with tab3:
    st.subheader("Diagnostic Comparison: Positive Case vs. Normal Baseline")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Case 1: Positive Pneumonia (`Target = 1`)")
        if pos_img_path.exists():
            st.image(Image.open(pos_img_path), use_container_width=True)
    with c2:
        st.markdown("#### Case 2: Healthy Baseline (`Target = 0`)")
        if neg_img_path.exists():
            st.image(Image.open(neg_img_path), use_container_width=True)

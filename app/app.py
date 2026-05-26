# =========================================================
# 🫁 PulmoTrack AI - Chest X-Ray Pneumonia Detection
# =========================================================

import os
import sys
import cv2
import numpy as np
import streamlit as st

# =========================================================
# PROJECT PATH SETUP
# =========================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from prediction.predictor import PneumoniaInferenceEngine

# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PulmoTrack AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1, h2, h3 {
    color: #12344d;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid #0d6efd;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.metric-title {
    font-size: 14px;
    color: #6c757d;
    font-weight: 600;
    text-transform: uppercase;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #111;
    margin-top: 5px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    model_path = os.path.join(
        BASE_DIR,
        "Models",
        "efficientnet_chest_xray_model.keras"
    )

    return PneumoniaInferenceEngine(model_path=model_path)


try:
    engine = load_model()
    st.sidebar.success("✅ Model Loaded Successfully")

except Exception as e:
    st.sidebar.error(f"❌ Failed to Load Model\n\n{e}")
    st.stop()

# =========================================================
# PAGE HEADER
# =========================================================

st.title("🫁 PulmoTrack AI")
st.caption("AI-powered Chest X-Ray Pneumonia Detection System")

st.divider()

# =========================================================
# SIDEBAR - SAMPLE IMAGES
# =========================================================

st.sidebar.header("📁 Sample Images")

sample_dir = os.path.join(BASE_DIR, "sample_data")

sample_files = []

if os.path.exists(sample_dir):
    sample_files = [
        file for file in os.listdir(sample_dir)
        if file.lower().endswith(("png", "jpg", "jpeg"))
    ]

selected_sample = None

if sample_files:

    selected_file = st.sidebar.selectbox(
        "Choose Sample Image",
        ["None"] + sample_files
    )

    if selected_file != "None":
        selected_sample = os.path.join(sample_dir, selected_file)

# =========================================================
# BATCH PREDICTION
# =========================================================

st.sidebar.divider()

st.sidebar.header("📊 Batch Prediction")

if st.sidebar.button("Run Batch Analysis"):

    if sample_files:

        image_paths = [
            os.path.join(sample_dir, file)
            for file in sample_files
        ]

        with st.spinner("Running predictions..."):

            batch_results = engine.predict_batch_directory(image_paths)

        st.subheader("Batch Prediction Results")
        st.dataframe(batch_results, use_container_width=True)

    else:
        st.sidebar.warning("No sample images found.")

# =========================================================
# IMAGE INPUT SECTION
# =========================================================

st.subheader("📤 Upload Chest X-Ray")

uploaded_file = st.file_uploader(
    "Upload PNG or JPEG Image",
    type=["png", "jpg", "jpeg"]
)

image = None
caption = ""

# Uploaded image
if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(file_bytes, 1)
    caption = "Uploaded Image"

# Sample image
elif selected_sample is not None:

    image = cv2.imread(selected_sample)
    caption = f"Sample Image - {os.path.basename(selected_sample)}"

# =========================================================
# PREDICTION SECTION
# =========================================================

if image is not None:

    col1, col2 = st.columns(2, gap="large")

    # -----------------------------------------------------
    # IMAGE DISPLAY
    # -----------------------------------------------------

    with col1:

        st.subheader("🖼️ Chest X-Ray")

        st.image(
            image,
            channels="BGR",
            caption=caption,
            use_container_width=True
        )

    # -----------------------------------------------------
    # PREDICTION RESULTS
    # -----------------------------------------------------

    with col2:

        st.subheader("🧠 Prediction")

        with st.spinner("Running prediction..."):

            label, confidence = engine.predict_single_vector(image)

        # Prediction Cards
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Prediction</div>
            <div class="metric-value">{label}</div>
        </div>

        <div class="metric-card">
            <div class="metric-title">Confidence</div>
            <div class="metric-value">{confidence:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # -------------------------------------------------
        # RECOMMENDATIONS
        # -------------------------------------------------

        st.subheader("📋 Clinical Notes")

        if label == "PNEUMONIA":

            st.error(
                "Possible pneumonia detected in the chest X-ray."
            )

            st.markdown("""
            - Recommend radiologist review
            - Correlate with patient symptoms
            - Consider additional diagnostic testing
            - Mark case as high priority
            """)

        else:

            st.success(
                "No major pneumonia patterns detected."
            )

            st.markdown("""
            - Appears normal based on model prediction
            - Continue routine clinical evaluation
            - Monitor symptoms if they persist
            """)

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "This application is intended for educational and portfolio purposes only. "
    "It should not be used as a replacement for professional medical diagnosis."
)
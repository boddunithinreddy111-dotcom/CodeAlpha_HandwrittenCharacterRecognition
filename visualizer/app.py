"""
Streamlit Web Application for Handwritten Character Recognition.
CodeAlpha ML Internship - Task 3.
Features Interactive Drawing Canvas, Image Upload, and Real-Time Deep Learning Inference.
Run locally with: streamlit run visualizer/app.py
"""

import sys
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_character_image, load_character_model
from src.image_processing import preprocess_image_array, array_to_image_preview

# Optional drawing canvas component
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

st.set_page_config(
    page_title="Handwritten Character Recognition | CodeAlpha Task 3",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .pred-card {
        background-color: #EEF2FF;
        border-left: 6px solid #4F46E5;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .pred-char {
        font-size: 3.5rem;
        font-weight: 800;
        color: #1E40AF;
        line-height: 1.1;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">✍️ Intelligent Handwritten Character Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">CodeAlpha Machine Learning Internship — Task 3: Image Processing & Deep Learning</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🛠️ Input Mode Selection")
mode = st.sidebar.radio(
    "Choose Input Method:",
    ["✏️ Draw Character on Canvas", "📁 Upload Image File", "🖼️ Showcase Sample Gallery"]
)

st.sidebar.write("---")
st.sidebar.markdown("""
**Model Details:**
- **Architecture**: Deep Neural Network / CNN
- **Resolution**: 28x28 Grayscale Normalized
- **Classes**: 26 Alphabet Categories (A - Z)
- **Features**: Center-of-Mass Alignment & Adaptive Denoising
""")

col_left, col_right = st.columns([1.1, 0.9])
input_image = None

with col_left:
    if mode == "✏️ Draw Character on Canvas":
        st.subheader("Interactive Drawing Canvas")
        st.write("Draw a single uppercase letter (A-Z) in the box below using your mouse or touchpad:")

        if CANVAS_AVAILABLE:
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=18,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                return_image_data=True,
                key="canvas",
            )
            if canvas_result is not None:
                try:
                    raw_data = canvas_result.image_data
                    if raw_data is not None and np.max(raw_data[:, :, :3]) > 10:
                        input_image = raw_data
                except Exception:
                    pass
        else:
            st.info("ℹ️ `streamlit-drawable-canvas` is not installed yet. You can use the **'Upload Image File'** or **'Showcase Sample Gallery'** options, or install it via `pip install streamlit-drawable-canvas`.")

    elif mode == "📁 Upload Image File":
        st.subheader("Upload Handwritten Character Image")
        uploaded_file = st.file_uploader(
            "Upload any handwritten letter image (.png, .jpg, .jpeg):",
            type=["png", "jpg", "jpeg"]
        )
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Uploaded Character Image", width=180)

    elif mode == "🖼️ Showcase Sample Gallery":
        st.subheader("Select from Showcase Test Samples")
        samples_dir = PROJECT_ROOT / "data" / "samples"
        sample_files = list(samples_dir.glob("sample_*.png")) if samples_dir.exists() else []

        if sample_files:
            char_options = [f.stem.replace("sample_", "") for f in sample_files]
            selected_char = st.selectbox("Choose a sample letter to test:", char_options)
            chosen_file = samples_dir / f"sample_{selected_char}.png"
            input_image = Image.open(chosen_file)
            st.image(input_image, caption=f"Selected Sample: '{selected_char}'", width=180)
        else:
            st.warning("Sample images not found. Run `python src/train.py` first to generate sample images.")

    predict_btn = st.button("🚀 Recognize Character", type="primary", use_container_width=True)

with col_right:
    st.subheader("Deep Learning Prediction")

    if predict_btn and input_image is not None:
        try:
            with st.spinner("Processing image and running Deep Learning inference..."):
                result = predict_character_image(input_image)

            predicted_char = result["predicted_character"]
            confidence = result["confidence"]
            top_preds = result["top_predictions"]
            preview_img = result["preprocessed_image"]

            st.markdown(f"""
            <div class="pred-card">
                <div>PREDICTED CHARACTER</div>
                <div class="pred-char">{predicted_char}</div>
                <div>Model Confidence: <strong>{confidence}%</strong></div>
            </div>
            """, unsafe_allow_html=True)

            p1, p2 = st.columns([1, 1])
            with p1:
                st.write("**Preprocessed 28x28 Tensor:**")
                st.image(preview_img.resize((140, 140), Image.Resampling.NEAREST), caption="Normalized & Centered")

            with p2:
                st.write("**Top Candidates:**")
                top_df = pd.DataFrame(top_preds)[["character", "confidence"]]
                top_df.columns = ["Alphabet", "Confidence (%)"]
                st.dataframe(top_df, hide_index=True, use_container_width=True)

            st.write("---")
            st.write("**Class Probability Distribution:**")
            chart_data = pd.DataFrame(top_preds).set_index("character")["probability"]
            st.bar_chart(chart_data)

        except FileNotFoundError as fnf:
            st.warning(f"⚠️ {fnf}")
        except Exception as e:
            st.error(f"Error during recognition: {e}")
    elif predict_btn and input_image is None:
        st.info("👈 Please draw or upload a character first, then click 'Recognize Character'.")
    else:
        st.info("👈 Provide an input using the left panel and click **'Recognize Character'** to evaluate.")

st.write("---")
st.markdown("""
**CodeAlpha Task 3 Submission Notes:**
- **Objective:** Recognize handwritten characters/alphabets.
- **Approach:** Image Processing (Inversion, Center of Mass Alignment, Scaling) + Deep Neural Network / CNN.
- **Dataset:** EMNIST / A-Z Handwritten Alphabets.
""")

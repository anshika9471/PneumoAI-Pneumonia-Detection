
from pathlib import Path

import streamlit as st
import tensorflow as tf
from PIL import Image
from datetime import datetime
import io

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="PneumoAI Dashboard",
    page_icon="🫁",
    layout="wide"
)

# ---------------- MODEL SETTINGS ----------------

MODEL_PATH = Path(__file__).parent / "models" / "pneumonia_mobilenetv2.keras"

IMG_SIZE = (224, 224)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
THRESHOLD = 0.5


# ---------------- SESSION STATE ----------------

if "history" not in st.session_state:
    st.session_state.history = []

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "uploaded_file_id" not in st.session_state:
    st.session_state.uploaded_file_id = None


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.stApp {
    background-color: #0b1220;
    color: #ffffff;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #38bdf8;
    margin-bottom: 5px;
}

.subtitle {
    color: #a7b5ca;
    font-size: 16px;
    margin-bottom: 25px;
}

.section-card {
    background-color: #172338;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #263b56;
    margin-bottom: 15px;
}

.result-normal {
    background-color: #123c35;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #20c997;
    text-align: center;
}

.result-pneumonia {
    background-color: #48252c;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #ff6b6b;
    text-align: center;
}

.metric-card {
    background-color: #172338;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #263b56;
    text-align: center;
}

.metric-value {
    font-size: 25px;
    font-weight: bold;
    color: #38bdf8;
}

.metric-label {
    color: #a7b5ca;
    font-size: 14px;
}

div.stButton > button {
    border-radius: 10px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ---------------- MODEL LOADING ----------------

@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# ---------------- IMAGE PREPROCESSING ----------------

def preprocess(image_bytes):

    image = tf.image.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False
    )

    image = tf.image.resize(image, IMG_SIZE)

    image = tf.cast(image, tf.float32)

    image = tf.expand_dims(image, axis=0)

    return image


# ---------------- PREDICTION FUNCTION ----------------

def predict(model, image_bytes):

    image = preprocess(image_bytes)

    probability = float(
        model(image, training=False).numpy()[0][0]
    )

    index = int(probability >= THRESHOLD)

    prediction = CLASS_NAMES[index]

    confidence = (
        probability if index == 1
        else 1 - probability
    )

    return prediction, confidence, probability


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("🫁 PneumoAI")

    st.markdown("---")

    st.subheader("About Project")

    st.write(
        "PneumoAI is a CNN-based educational project "
        "that classifies chest X-ray images into "
        "NORMAL and PNEUMONIA categories."
    )

    st.markdown("---")

    st.subheader("Model Information")

    st.write("**Architecture:** MobileNetV2")

    st.write("**Technique:** Transfer Learning")

    st.write("**Image Size:** 224 × 224")

    st.write("**Classes:** Normal / Pneumonia")

    st.markdown("---")

    st.subheader("Important Notice")

    st.warning(
        "This application is only for educational "
        "demonstration. It is not a medical device "
        "and must not be used for diagnosis or "
        "treatment decisions."
    )


# ---------------- MAIN HEADER ----------------

st.markdown(
    '<div class="main-title">🫁 PneumoAI Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered Pneumonia Classification from Chest X-Ray Images'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Upload a chest X-ray image to see the model's prediction. "
    "This is an educational demonstration only."
)


# ---------------- MODEL CHECK ----------------

if not MODEL_PATH.exists():

    st.error(
        "Model file not found! Please check "
        "models/pneumonia_mobilenetv2.keras"
    )

    st.stop()


# Load model

model = load_model()


# ---------------- UPLOAD SECTION ----------------

st.markdown("## 📤 Upload Chest X-Ray")

col_upload, col_info = st.columns([1.5, 1])

with col_upload:

    st.markdown("""
    <div class="section-card">
        <h3>📁 Upload Your Image</h3>
        <p>Select a chest X-ray image in JPG, JPEG, PNG or BMP format.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose X-ray image",
        type=["jpg", "jpeg", "png", "bmp"],
        key=f"xray_uploader_{st.session_state.uploader_key}"
    )


with col_info:

    st.markdown("""
    <div class="section-card">
        <h3>📋 Instructions</h3>
        <p>1. Select a chest X-ray image.</p>
        <p>2. Wait for model processing.</p>
        <p>3. View the prediction and scores.</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- CLEAR IMAGE BUTTON ----------------

if uploaded is not None:

    if st.button("🗑️ Clear Uploaded Image"):

        st.session_state.uploaded_file_id = None

        st.session_state.uploader_key += 1

        st.rerun()


# ---------------- PREDICTION SECTION ----------------

if uploaded is not None:

    image_bytes = uploaded.getvalue()

    file_id = (
        f"{st.session_state.uploader_key}_"
        f"{uploaded.name}_{uploaded.size}"
    )

    st.markdown("---")

    st.markdown("## 🔍 X-Ray Analysis")

    col_image, col_result = st.columns([1, 1])

    # Image preview

    with col_image:

        st.markdown(
            '<div class="section-card">'
            '<h3>Uploaded X-Ray</h3>'
            '</div>',
            unsafe_allow_html=True
        )

        st.image(
            Image.open(io.BytesIO(image_bytes)).convert("RGB"),
            caption=uploaded.name,
            use_container_width=True
        )


    # Prediction

    with st.spinner("AI model is analyzing the image..."):

        prediction, confidence, probability = predict(
            model,
            image_bytes
        )


    with col_result:

        st.markdown(
            '<div class="section-card">'
            '<h3>🤖 Prediction Result</h3>'
            '</div>',
            unsafe_allow_html=True
        )

        if prediction == "NORMAL":

            st.markdown(
                f"""
                <div class="result-normal">
                    <h2>✅ NORMAL</h2>
                    <h3>{confidence * 100:.2f}% Confidence</h3>
                    <p>The model classified this image as NORMAL.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-pneumonia">
                    <h2>⚠️ PNEUMONIA</h2>
                    <h3>{confidence * 100:.2f}% Confidence</h3>
                    <p>The model classified this image as PNEUMONIA.</p>
                </div>
                """,
                unsafe_allow_html=True
            )


    # Probability scores

    st.markdown("### 📊 Prediction Scores")

    score_col1, score_col2 = st.columns(2)

    with score_col1:

        normal_score = (1 - probability) * 100

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{normal_score:.2f}%</div>
                <div class="metric-label">NORMAL Score</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(float(1 - probability))


    with score_col2:

        pneumonia_score = probability * 100

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{pneumonia_score:.2f}%</div>
                <div class="metric-label">PNEUMONIA Score</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(float(probability))


    # ---------------- SAVE HISTORY ----------------

    if st.session_state.uploaded_file_id != file_id:

        st.session_state.history.insert(
            0,
            {
                "Image": uploaded.name,
                "Prediction": prediction,
                "Confidence": f"{confidence * 100:.2f}%",
                "Time": datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")
            }
        )

        st.session_state.uploaded_file_id = file_id


# ---------------- PREDICTION HISTORY ----------------

st.markdown("---")

st.markdown("## 📜 Prediction History")

if st.session_state.history:

    st.write(
        f"Total Predictions: {len(st.session_state.history)}"
    )

    st.dataframe(
        st.session_state.history,
        use_container_width=True,
        hide_index=True
    )

    if st.button("🧹 Clear Prediction History"):

        st.session_state.history = []

        st.session_state.uploaded_file_id = None

        st.rerun()

else:

    st.info("No prediction history available yet.")


# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#94a3b8;">
        <p><b>PneumoAI</b> | Pneumonia X-Ray Classification Demo</p>
        <p>B.Tech Semester Project | Deep Learning</p>
        <p>For educational purposes only — not for medical diagnosis.</p>
    </div>
    """,
    unsafe_allow_html=True
)

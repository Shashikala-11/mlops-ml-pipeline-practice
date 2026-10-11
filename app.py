
import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"

# Add the real vectorizer path here if your pipeline saves one.
VECTORIZER_CANDIDATES = [
    BASE_DIR / "models" / "tfidf_vectorizer.pkl",
    BASE_DIR / "models" / "vectorizer.pkl",
    BASE_DIR / "models" / "tfidf.pkl",
]

st.set_page_config(
    page_title="SpamShield | SMS Classifier",
    page_icon="🛡️",
    layout="wide",
)

# --------------------------------------------------
# UI styling
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: #f3f7fc;
}
.block-container {
    padding-top: 2rem;
    max-width: 1150px;
}
.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(120deg, #102c4a, #176b78);
    color: white;
    margin-bottom: 24px;
}
.hero h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 8px;
}
.hero p {
    color: #e1f5f5;
    font-size: 17px;
}
.metric-card {
    background: white;
    padding: 18px;
    border: 1px solid #dce6ef;
    border-radius: 14px;
}
div.stButton > button {
    background: #087f8c;
    color: white;
    border-radius: 10px;
    border: 0;
    min-height: 48px;
    font-weight: 700;
}
div.stButton > button:hover {
    background: #066773;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Load existing artifacts
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    return model


@st.cache_resource
def load_vectorizer(path_string):
    with open(path_string, "rb") as file:
        return pickle.load(file)


def load_metrics():
    if not METRICS_PATH.exists():
        return {}

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return {}


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🛡️ SpamShield</h1>
    <p>Machine Learning SMS Spam Detection System</p>
    <p>Analyze a message using your trained Random Forest model.</p>
</div>
""", unsafe_allow_html=True)

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

metrics = load_metrics()

# --------------------------------------------------
# Model overview
# --------------------------------------------------
st.subheader("📊 Model Overview")

m1, m2, m3 = st.columns(3)

m1.metric("Model", "Random Forest")
m2.metric(
    "Model artifact",
    "Available" if MODEL_PATH.exists() else "Missing",
)

accuracy = None
for key in ("accuracy", "Accuracy", "test_accuracy"):
    if key in metrics and isinstance(metrics[key], (int, float)):
        accuracy = metrics[key]
        break

if accuracy is not None:
    if accuracy <= 1:
        accuracy *= 100
    m3.metric("Saved test accuracy", f"{accuracy:.2f}%")
else:
    m3.metric("Saved test accuracy", "See report")

st.caption(
    "Saved metrics are shown as recorded by your pipeline. "
    "They are not recalculated by this dashboard."
)

st.divider()

# --------------------------------------------------
# Prediction area
# --------------------------------------------------
left, right = st.columns([1.25, 0.75], gap="large")

with left:
    st.subheader("✉️ Analyze an SMS")

    message = st.text_area(
        "Enter your message",
        placeholder=(
            "Example: Congratulations! You have won a prize. "
            "Call now to claim your reward."
        ),
        height=170,
    )

    classify = st.button(
        "🔍 Detect Spam or Ham",
        use_container_width=True,
    )

    if classify:
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            vectorizer_path = next(
                (
                    path for path in VECTORIZER_CANDIDATES
                    if path.exists()
                ),
                None,
            )

            if vectorizer_path is None:
                st.error(
                    "The trained Random Forest was found, but its "
                    "fitted TF-IDF vectorizer was not found in the "
                    "expected models/ locations. Raw SMS text cannot "
                    "be passed directly to this model."
                )
                st.info(
                    "Save the fitted vectorizer from your existing "
                    "feature-engineering stage, then add its path to "
                    "VECTORIZER_CANDIDATES near the top of app.py."
                )
            else:
                try:
                    vectorizer = load_vectorizer(
                        str(vectorizer_path)
                    )
                    features = vectorizer.transform([message])
                    prediction = model.predict(features)[0]

                    probabilities = None
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(features)[0]

                    st.session_state["last_prediction"] = str(
                        prediction
                    )
                    st.session_state["last_message"] = message
                    st.session_state["last_probabilities"] = (
                        probabilities.tolist()
                        if probabilities is not None else None
                    )

                except Exception as error:
                    st.error(f"Prediction failed: {error}")

    if "last_prediction" in st.session_state:
        prediction = st.session_state["last_prediction"]
        normalized = prediction.strip().lower()

        st.markdown("### Prediction result")

        if normalized in {"spam", "1"}:
            st.error("🚨 SPAM MESSAGE")
        elif normalized in {"ham", "0", "not spam", "legitimate"}:
            st.success("✅ HAM — LEGITIMATE MESSAGE")
        else:
            st.info(f"Model prediction: {prediction}")

        probs = st.session_state.get("last_probabilities")
        if probs is not None:
            st.caption(
                "Model probability estimates (not a guarantee "
                "of correctness):"
            )
            classes = getattr(model, "classes_", [])
            for label, probability in zip(classes, probs):
                st.write(f"{label}: {probability:.1%}")
                st.progress(float(probability))

with right:
    st.subheader("🧠 How it works")

    st.markdown("""
    **1. Message input**

    Enter the SMS text to analyze.

    **2. Text transformation**

    The fitted TF-IDF vectorizer converts text into the features
    expected by the trained model.

    **3. Random Forest**

    The existing classifier predicts the message category.

    **4. Result**

    The interface displays the prediction and probability
    estimates when available.
    """)

    st.info(
        "This dashboard reuses your existing model artifact. "
        "It does not rerun or replace your DVC pipeline."
    )

# --------------------------------------------------
# Saved evaluation metrics
# --------------------------------------------------
st.divider()
st.subheader("📈 Pipeline Evaluation Metrics")

if metrics:
    st.json(metrics)
else:
    st.warning(
        "reports/metrics.json was not found or contains no "
        "readable metrics."
    )

with st.expander("Project and pipeline details"):
    st.write(f"**Model path:** `{MODEL_PATH}`")
    st.write(f"**Metrics path:** `{METRICS_PATH}`")
    st.write("**Pipeline manager:** DVC")
    st.write("**Classifier:** Random Forest")

st.caption(
    "Educational demonstration only. Predictions may be wrong; "
    "review messages before taking action."
)
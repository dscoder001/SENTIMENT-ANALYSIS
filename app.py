"""
Restaurant Review Sentiment Analysis — Streamlit web app.

Loads the pre-trained TF-IDF + SVC pipeline and classifies any
review as Positive or Negative, with a model confidence score.
"""

from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path("model/sentiment_pipeline.joblib")
LABELS = {0: "Negative", 1: "Positive"}

EXAMPLES = [
    "The food was absolutely delicious and the service was fantastic!",
    "Worst experience ever. Cold food and rude staff.",
    "It was okay, nothing special but not terrible either.",
]

st.set_page_config(page_title="Sentiment Analysis", page_icon="🍽️", layout="centered")


@st.cache_resource
def load_model():
    """Load the serialized pipeline once and cache it across reruns."""
    return joblib.load(MODEL_PATH)


def predict(model, text: str):
    label = int(model.predict([text])[0])
    proba = model.predict_proba([text])[0]
    confidence = float(proba[label])
    return label, confidence


# ---- session state for example buttons -------------------------------------
if "review_text" not in st.session_state:
    st.session_state.review_text = ""


def set_example(text: str) -> None:
    st.session_state.review_text = text


# ---- UI --------------------------------------------------------------------
st.title("🍽️ Restaurant Review Sentiment Analysis")
st.caption("TF-IDF + Support Vector Classifier · ~82.7% test accuracy")

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model artifact not found. Run `python train.py` first "
        "(this happens automatically inside the Docker build)."
    )
    st.stop()

st.write("Type a restaurant review below, or try one of the examples.")

cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    short = example[:24] + "…" if len(example) > 24 else example
    col.button(short, on_click=set_example, args=(example,), use_container_width=True)

review = st.text_area(
    "Review text",
    key="review_text",
    height=120,
    placeholder="e.g. The pasta was incredible and the staff were so friendly!",
)

if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        label, confidence = predict(model, review)
        sentiment = LABELS[label]
        if label == 1:
            st.success(f"### Positive  ·  {confidence:.0%} confidence")
        else:
            st.error(f"### Negative  ·  {confidence:.0%} confidence")
        st.progress(confidence)

st.divider()
st.caption(
    "Model: TfidfVectorizer → SVC pipeline (scikit-learn). "
    "Trained on 1,000 labelled restaurant reviews."
)

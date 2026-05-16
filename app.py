import streamlit as st
import pickle
import numpy as np
import re
import string
import json

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)

# ── Load artifacts ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("best_model.pkl",        "rb") as f: model = pickle.load(f)
    with open("tfidf_vectorizer.pkl",  "rb") as f: tfidf = pickle.load(f)
    with open("label_encoder.pkl",     "rb") as f: le    = pickle.load(f)
    with open("model_metadata.json",   "r")  as f: meta  = json.load(f)
    return model, tfidf, le, meta

model, tfidf, le, meta = load_artifacts()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english")) - {"no","not","nor","never"}

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)

LABEL_EMOJI = {
    "gender"             : "⚧️  Gender-based",
    "religion"           : "🕌  Religion-based",
    "age"                : "👴  Age-based",
    "ethnicity"          : "🌍  Ethnicity-based",
    "not_cyberbullying"  : "✅  Not Cyberbullying",
    "other_cyberbullying": "⚠️  Other Cyberbullying",
}
LABEL_COLOR = {
    "gender"             : "#e74c3c",
    "religion"           : "#8e44ad",
    "age"                : "#e67e22",
    "ethnicity"          : "#c0392b",
    "not_cyberbullying"  : "#27ae60",
    "other_cyberbullying": "#f39c12",
}

# ── UI ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cyberbullying Detector", page_icon="🛡️", layout="wide")
st.title("🛡️ Cyberbullying Tweet Detector")
st.markdown(
    f"> **Model:** `{meta['best_model_name']}` &nbsp;|&nbsp;"
    f"**Accuracy:** `{meta['accuracy']*100:.2f}%` &nbsp;|&nbsp;"
    f"**F1 Macro:** `{meta['f1_macro']:.4f}`"
)
st.markdown("---")

col1, col2 = st.columns([3, 2])

with col1:
    tweet_input = st.text_area(
        "📝 Enter tweet text:",
        height=160,
        placeholder="Type or paste a tweet here..."
    )
    analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")

with col2:
    st.markdown("**ℹ️ About the model**")
    st.json({
        "Model"       : meta["best_model_name"],
        "Best Params" : meta["best_params"],
        "Accuracy"    : f"{meta['accuracy']*100:.2f}%",
        "F1 Macro"    : f"{meta['f1_macro']:.4f}",
    })

if analyze_btn:
    if not tweet_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        cleaned  = preprocess_text(tweet_input)
        vec      = tfidf.transform([cleaned])
        pred_idx = model.predict(vec)[0]
        label    = le.inverse_transform([pred_idx])[0]
        color    = LABEL_COLOR.get(label, "#888")
        emoji_lbl= LABEL_EMOJI.get(label, label)

        st.markdown("---")
        st.markdown(f"### Prediction")
        st.markdown(
            f"<div style='background:{color};padding:16px;border-radius:10px;"
            f"color:white;font-size:22px;font-weight:bold;text-align:center;'>"
            f"{emoji_lbl}</div>", unsafe_allow_html=True
        )

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vec)[0]
            proba_dict = dict(sorted(
                zip(le.classes_, proba), key=lambda x: x[1], reverse=True
            ))
            st.markdown("#### Class Probabilities")
            for cls, prob in proba_dict.items():
                st.progress(float(prob), text=f"{LABEL_EMOJI.get(cls, cls)}: {prob*100:.1f}%")

        with st.expander("🔎 Cleaned text used for prediction"):
            st.code(cleaned)

st.markdown("---")
st.markdown("*Built with sklearn · TF-IDF · SMOTE · Streamlit*")
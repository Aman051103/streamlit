import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_resources():
    tokenizer_path = BASE_DIR / "tokenizer.pkl"
    model_path = BASE_DIR / "nexpree1.keras"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    model = load_model(model_path)
    return tokenizer, model
# Load tokenizer and model

tokenizer, model = load_resources()
max_len = 107  # from your README

def predict_next_words(text, top_k=4):
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding="pre")
    preds = model.predict(seq)[0]
    top_indices = preds.argsort()[-top_k:][::-1]
    inv_vocab = {v: k for k, v in tokenizer.word_index.items()}
    return [(inv_vocab.get(i, ""), float(preds[i])) for i in top_indices]

st.title("Smart Keyboard – Next Word Predictor")

user_input = st.text_input("Type your sentence:", "")

if user_input.strip():
    suggestions = predict_next_words(user_input)
    st.subheader("Suggestions")
    for word, prob in suggestions:
        if word:
            st.write(f"- **{word}** ({prob:.3f})")
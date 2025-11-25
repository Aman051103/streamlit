import streamlit as st
import numpy as np
import pickle
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = Path(__file__).resolve().parent
MAX_LEN = 107  # from your README

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

tokenizer, model = load_resources()

def predict_next_words(text, top_k=4):
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=MAX_LEN, padding="pre")
    preds = model.predict(seq)[0]
    top_indices = preds.argsort()[-top_k:][::-1]
    inv_vocab = {v: k for k, v in tokenizer.word_index.items()}
    return [inv_vocab.get(i, "") for i in top_indices if inv_vocab.get(i, "")]

st.title("Smart Keyboard – Next Word Predictor")

# --- Session state setup ---
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

# Text input bound directly to session state
text = st.text_input(
    "Type your sentence:",
    value=st.session_state.text_input,
    key="text_input",
)

# Always work from the latest text in session_state
current_text = st.session_state.text_input

# --- Generate suggestions whenever last char is space ---
suggestions = []
if current_text and current_text[-1].isspace():
    suggestions = predict_next_words(current_text)

if suggestions:
    st.subheader("Suggestions")
    cols = st.columns(len(suggestions))
    for i, word in enumerate(suggestions):
        if cols[i].button(word, key=f"suggestion_{i}"):
            # 1) Append chosen word + space
            new_text = (current_text or "") + word + " "
            # 2) Update the text input's state
            st.session_state.text_input = new_text
            # 3) Force an immediate rerun so new suggestions appear
            st.rerun()

st.write("Current text:")
st.write(st.session_state.text_input)
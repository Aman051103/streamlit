import streamlit as st
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Optional
import pandas as pd
import altair as alt

# --------------------------
# CONFIG (same as your script)
# --------------------------

WORKSPACE = Path(__file__).resolve().parent
MODEL_PATH = WORKSPACE / "autoencoder.keras"

TRAIN_MIN = -7.0903741
TRAIN_MAX = 7.4021031
RECONSTRUCTION_THRESHOLD = 0.03392139


# --------------------------
# MODEL ARCHITECTURE (same as before)
# --------------------------

@tf.keras.utils.register_keras_serializable()
class AnomalyDetector(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(8, activation="relu"),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(140, activation="sigmoid"),
            ]
        )

    def call(self, inputs):
        encoded = self.encoder(inputs)
        return self.decoder(encoded)


# --------------------------
# UTILITY FUNCTIONS
# --------------------------

def load_autoencoder(model_path: Path) -> tf.keras.Model:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(
        model_path, custom_objects={"AnomalyDetector": AnomalyDetector}
    )


def resample_signal(signal: np.ndarray, target_length: int = 140) -> np.ndarray:
    if signal.size == target_length:
        return signal
    original_positions = np.linspace(0.0, 1.0, num=signal.size)
    target_positions = np.linspace(0.0, 1.0, num=target_length)
    return np.interp(target_positions, original_positions, signal).astype(np.float32)


def min_max_scale(signal: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    return ((signal - min_val) / (max_val - min_val)).astype(np.float32)


def score_signal(model, normalized_signal):
    tensor = tf.convert_to_tensor(normalized_signal[None, :], dtype=tf.float32)
    reconstruction = model(tensor)
    loss = tf.keras.losses.mae(reconstruction, tensor)
    return float(loss.numpy().squeeze()), reconstruction.numpy().squeeze()


def classify(loss: float, threshold: float = RECONSTRUCTION_THRESHOLD) -> str:
    return "NORMAL" if loss < threshold else "ANOMALOUS"


def parse_uploaded_file(file) -> Optional[np.ndarray]:
    """Reads CSV, TXT, TSV, or newline-separated numbers."""
    try:
        df = pd.read_csv(file, header=None)
        arr = df.values.flatten().astype(np.float32)
        return arr
    except Exception:
        try:
            raw = file.read().decode()
            nums = [float(x) for x in raw.replace(",", " ").split()]
            return np.array(nums, dtype=np.float32)
        except Exception:
            return None


# --------------------------
# STREAMLIT UI
# --------------------------

st.set_page_config(page_title="ECG Anomaly Detector", layout="wide")
st.title("🫀 ECG Anomaly Detection (Autoencoder)")

st.write("Upload an ECG signal file (CSV, TXT, TSV, or newline-separated values).")

uploaded_file = st.file_uploader("Upload ECG Data File", type=["csv", "txt", "tsv"])

if uploaded_file:
    signal = parse_uploaded_file(uploaded_file)

    if signal is None:
        st.error("❌ Unable to parse numeric ECG signal from file.")
        st.stop()

    st.success(f"Loaded signal with {len(signal)} samples.")

    st.line_chart(signal, height=200)

    if st.button("Run Detection"):
        with st.spinner("Loading model..."):
            model = load_autoencoder(MODEL_PATH)

        with st.spinner("Processing ECG…"):
            resampled = resample_signal(signal, 140)
            normalized = min_max_scale(resampled, TRAIN_MIN, TRAIN_MAX)
            loss, reconstruction = score_signal(model, normalized)
            verdict = classify(loss)

        # Results
        st.subheader("🔍 Detection Result")
        st.write(f"**Reconstruction MAE:** `{loss:.6f}`")
        st.write(f"**Threshold:** `{RECONSTRUCTION_THRESHOLD:.6f}`")
        st.write(f"### 🟢 NORMAL" if verdict == "NORMAL" else "### 🔴 ANOMALOUS")

        # Plot reconstructed vs original
        plot_df = pd.DataFrame({
            "Original": resampled,
            "Reconstructed": reconstruction
        })

        st.subheader("📊 Original vs Reconstructed Signal")
        st.line_chart(plot_df)

else:
    st.info("Awaiting ECG file upload…")

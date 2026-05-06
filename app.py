"""
F-16 Maneuver Recognition — Streamlit Demo

Usage:
    streamlit run app.py
"""

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cnn import modul
from Dataset_corrected import F16FlightDatasetCorrected

# ── Constants ──────────────────────────────────────────────────────────────────
FEATURE_NAMES = ["Longitude", "Latitude", "Altitude", "Roll",
                 "Pitch", "Yaw", "Roll Rate", "Feature8"]

VERTICAL_LABELS = ["Down", "Level", "Up"]

VERTICAL_GROUP = {
    "Descent": "Down",
    "Level Flight": "Level",
    "Roll Left": "Level",
    "Roll Right": "Level",
    "Turn Left": "Level",
    "Turn Left Descent": "Down",
    "Turn Left Up": "Up",
    "Turn Right": "Level",
    "Turn Right Descent": "Down",
    "Turn Right Up": "Up",
    "Up": "Up",
    "Vertical Turn Descent": "Down",
    "Vertical Turn Up": "Up",
}

MANEUVER_FILES = {
    "Up": "01up.txt",
    "Level Flight": "02Level Flight.txt",
    "Descent": "03Descent.txt",
    "Turn Right": "04Turn right.txt",
    "Turn Left": "05Turn left.txt",
    "Turn Right Up": "06Turn right up.txt",
    "Turn Right Descent": "07Turn right descent.txt",
    "Turn Left Up": "08Turn left up.txt",
    "Turn Left Descent": "09Turn left descent.txt",
    "Vertical Turn Up": "10Vertical turn up.txt",
    "Roll Right": "11Roll right.txt",
    "Roll Left": "12Roll left.txt",
    "Vertical Turn Descent": "13Vertical turn descetn.txt",
}

CLASS_NAMES = sorted(MANEUVER_FILES.keys())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "flight_data")
MODEL_PATH = os.path.join(BASE_DIR, "model_weight", "best_model.pth")

# ── Resource loading (cached once per session) ─────────────────────────────────
@st.cache_resource(show_spinner="Loading model and preprocessor (first run only)...")
def load_resources():
    # Re-fit scaler with the same pipeline used during training
    ds = F16FlightDatasetCorrected(
        data_folder=DATA_DIR,
        time_steps=10,
        features_per_step=8,
        windows=[8, 10],
        window_strides={8: 2, 10: 1},
        add_delta=True,
        normalize=True,
        train_ratio=0.8,
    )
    ds.load_data()
    ds.preprocess_data(split_mode="grouped_random", random_state=42)
    scaler = ds.scaler

    model = modul(
        num_classes=13,
        feature_dim=16,
        time_steps=10,
        transformer_dropout=0.35,
        classifier_dropout=0.5,
        aux_num_classes=3,
    )
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model, scaler


# ── Raw sequence loading (cached per maneuver) ─────────────────────────────────
@st.cache_data(show_spinner=False)
def load_sequences(maneuver_name: str):
    filepath = os.path.join(DATA_DIR, MANEUVER_FILES[maneuver_name])
    sequences = []
    expected_len = 1 + 10 * 8
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            vals = line.strip().split(",")
            if len(vals) < expected_len:
                continue
            try:
                arr = np.array(list(map(float, vals[1:expected_len])), dtype=np.float32)
                sequences.append(arr.reshape(10, 8))
            except ValueError:
                continue
    return sequences


# ── Preprocessing & inference ──────────────────────────────────────────────────
def preprocess(seq: np.ndarray, scaler) -> torch.Tensor:
    diff = np.diff(seq, axis=0)
    zeros = np.zeros((1, 8), dtype=np.float32)
    delta = np.vstack([zeros, diff])
    combined = np.concatenate([seq, delta], axis=1)   # (10, 16)
    normalized = scaler.transform(combined)            # (10, 16)
    # Model input shape: (B=1, C=1, F=16, T=10)
    return torch.FloatTensor(normalized.T).unsqueeze(0).unsqueeze(0)


def run_inference(model, tensor: torch.Tensor):
    se_weights: dict[str, np.ndarray] = {}

    def make_hook(key):
        def hook(module, inp, out):
            se_weights[key] = out.detach().cpu().numpy().flatten()
        return hook

    h1 = model.se_a.fc.register_forward_hook(make_hook("raw"))
    h2 = model.se_b.fc.register_forward_hook(make_hook("delta"))

    with torch.no_grad():
        logits, aux_logits = model(tensor, return_aux=True)

    h1.remove()
    h2.remove()

    probs = torch.softmax(logits, dim=1).squeeze().numpy()
    aux_probs = torch.softmax(aux_logits, dim=1).squeeze().numpy()
    return probs, aux_probs, se_weights


# ── Plotting helpers ───────────────────────────────────────────────────────────
def plot_sensor_data(seq: np.ndarray, title: str) -> plt.Figure:
    fig, axes = plt.subplots(4, 2, figsize=(8, 7))
    time_axis = np.arange(10)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
              "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    for i, (ax, fname, color) in enumerate(zip(axes.flat, FEATURE_NAMES, colors)):
        ax.plot(time_axis, seq[:, i], "o-", color=color, markersize=4, linewidth=1.5)
        ax.set_title(fname, fontsize=9, fontweight="bold")
        ax.set_xticks(time_axis)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Ground Truth: {title}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_confidence(probs: np.ndarray) -> plt.Figure:
    sorted_idx = np.argsort(probs)
    sorted_names = [CLASS_NAMES[i] for i in sorted_idx]
    sorted_probs = probs[sorted_idx] * 100

    colors = ["#e74c3c" if p == sorted_probs.max() else "#3498db" for p in sorted_probs]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(sorted_names, sorted_probs, color=colors)
    ax.set_xlabel("Confidence (%)")
    ax.set_xlim(0, 105)
    for bar, val in zip(bars, sorted_probs):
        if val > 1:
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=8)
    ax.set_title("Per-Class Confidence", fontsize=11)
    plt.tight_layout()
    return fig


def plot_aux(aux_probs: np.ndarray) -> plt.Figure:
    colors = ["#e74c3c" if p == aux_probs.max() else "#95a5a6" for p in aux_probs]
    fig, ax = plt.subplots(figsize=(4, 2.5))
    bars = ax.bar(VERTICAL_LABELS, aux_probs * 100, color=colors)
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, aux_probs * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                f"{val:.1f}%", ha="center", fontsize=9)
    ax.set_title("Vertical Trend (Auxiliary Task)", fontsize=10)
    plt.tight_layout()
    return fig


def plot_se_weights(se_weights: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 2.8))
    channels = np.arange(1, 17)

    ax1.bar(channels, se_weights.get("raw", np.zeros(16)), color="#2980b9")
    ax1.set_title("Raw Branch — SE Channel Weights", fontsize=9)
    ax1.set_xlabel("Conv Channel")
    ax1.set_ylabel("Attention Weight")
    ax1.set_xticks(channels)
    ax1.tick_params(labelsize=7)

    ax2.bar(channels, se_weights.get("delta", np.zeros(16)), color="#e67e22")
    ax2.set_title("Delta Branch — SE Channel Weights", fontsize=9)
    ax2.set_xlabel("Conv Channel")
    ax2.set_ylabel("Attention Weight")
    ax2.set_xticks(channels)
    ax2.tick_params(labelsize=7)

    plt.tight_layout()
    return fig


# ── App layout ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F-16 Maneuver Recognition",
    page_icon="✈",
    layout="wide",
)

st.title("F-16 Maneuver Recognition Demo")
st.caption("Dual-Branch CNN + SE Attention + Transformer | 13-class flight maneuver classifier")

model, scaler = load_resources()

# Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Selection")
    selected_maneuver = st.selectbox("Maneuver Class (Ground Truth)", list(MANEUVER_FILES.keys()))
    sequences = load_sequences(selected_maneuver)
    max_idx = min(len(sequences) - 1, 999)
    seq_idx = st.slider("Sample Index", 0, max_idx, 0)
    st.caption(f"Total samples in this file: **{len(sequences)}**")
    st.divider()
    st.markdown("**Vertical group mapping**")
    for name, group in VERTICAL_GROUP.items():
        icon = {"Up": "⬆", "Level": "➡", "Down": "⬇"}[group]
        st.markdown(f"- {name} → {icon} {group}")

selected_seq = sequences[seq_idx]

# Main layout ──────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Raw Sensor Data")
    st.pyplot(plot_sensor_data(selected_seq, selected_maneuver))
    plt.close("all")

with col_right:
    st.subheader("Model Inference")
    run_btn = st.button("Run Prediction", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Running inference..."):
            tensor = preprocess(selected_seq, scaler)
            probs, aux_probs, se_weights = run_inference(model, tensor)

        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx] * 100
        is_correct = pred_class == selected_maneuver

        # ── Result header
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Prediction", pred_class, f"{confidence:.1f}% confidence")
        with res_col2:
            if is_correct:
                st.success(f"Correct ✓  (Ground truth: {selected_maneuver})")
            else:
                st.error(f"Incorrect ✗  (Ground truth: {selected_maneuver})")

        st.divider()

        # ── Confidence chart
        st.markdown("**Per-Class Confidence**")
        st.pyplot(plot_confidence(probs))
        plt.close("all")

        # ── Auxiliary task
        st.markdown("**Auxiliary Task — Vertical Trend Prediction**")
        expected_vertical = VERTICAL_GROUP.get(selected_maneuver, "?")
        pred_vertical = VERTICAL_LABELS[int(np.argmax(aux_probs))]
        aux_correct = pred_vertical == expected_vertical
        st.caption(
            f"Expected: **{expected_vertical}** → Predicted: **{pred_vertical}** "
            + ("✓" if aux_correct else "✗")
        )
        st.pyplot(plot_aux(aux_probs))
        plt.close("all")

        # ── SE attention weights
        st.markdown("**SE Attention Weights (channel importance)**")
        st.caption("Higher = model relies more on that convolutional channel.")
        st.pyplot(plot_se_weights(se_weights))
        plt.close("all")

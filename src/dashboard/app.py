"""
Layer 6 — Dashboard
app.py

Main Streamlit application entry point.
Orchestrates all 6 pipeline layers into a live interactive dashboard.

Run with:
    streamlit run src/dashboard/app.py
"""

import sys
import os
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Explicitly load .env from project root
from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT / ".env", override=True)

import numpy as np
import streamlit as st

from src.dashboard.components import (
    set_page_config, inject_css, render_header,
    render_signal_metrics, render_fault_card,
    render_explanation, render_corrective_actions,
    render_report,
)
from src.dashboard.plots import (
    plot_waveform, plot_fft_spectrum, plot_spectrogram,
    plot_feature_radar, plot_harmonic_bars,
)
from src.dashboard.signal_input import render_signal_input
from src.dsp.fft_analyzer import compute_fft, fft_summary
from src.dsp.spectrogram import compute_spectrogram
from src.features.feature_pipeline import extract_features, extract_features_vector
from config import FAULT_TYPES, MODELS_DIR


# ── Page setup ────────────────────────────────────────────────────────────

set_page_config()
inject_css()


# ── Sidebar ───────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Minimal sidebar — just analysis toggles. Signal input is in main area."""
    st.sidebar.title("⚙️ Analysis Options")

    run_genai        = st.sidebar.checkbox("AI diagnostics (Groq)", value=True)
    show_spectrogram = st.sidebar.checkbox("Show spectrogram",      value=True)
    show_radar       = st.sidebar.checkbox("Show feature radar",    value=True)
    show_harmonics   = st.sidebar.checkbox("Show harmonic bars",    value=True)

    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["Analyse Signal", "Train Models"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Smart Grid Signal Intelligence\nPowered by Groq + LangChain")

    return {
        "run_genai":        run_genai,
        "show_spectrogram": show_spectrogram,
        "show_radar":       show_radar,
        "show_harmonics":   show_harmonics,
        "page":             page,
    }


# ── Model loader ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models...")
def load_models():
    """Load trained models — auto-trains if not found (first run)."""
    from src.ml.model_utils import load_model, load_scaler

    rf_path  = MODELS_DIR / "random_forest.pkl"
    iso_path = MODELS_DIR / "isolation_forest.pkl"

    if not rf_path.exists() or not iso_path.exists():
        st.info("First run — training models (~60 seconds)...")
        _auto_train()

    models = {}
    try:
        models["classifier"] = load_model("random_forest")
        models["clf_scaler"]  = load_scaler("scaler_random_forest")
    except FileNotFoundError:
        models["classifier"] = None
        models["clf_scaler"]  = None

    try:
        models["anomaly"]    = load_model("isolation_forest")
        models["ano_scaler"] = load_scaler("scaler_isolation_forest")
    except FileNotFoundError:
        models["anomaly"]    = None
        models["ano_scaler"] = None

    return models


def _auto_train():
    """Auto-train RF + Isolation Forest on first run."""
    from src.simulation.signal_generator import generate_dataset
    from src.features.feature_pipeline import build_feature_dataframe, save_features
    from src.ml.train_classifier import train_classifier
    from src.ml.anomaly_detector import train_anomaly_detector

    with st.spinner("Generating training data..."):
        X, y = generate_dataset(n_per_class=150, add_noise=True)
    with st.spinner("Extracting features..."):
        df = build_feature_dataframe(X, y)
        save_features(df)
    with st.spinner("Training Random Forest..."):
        train_classifier(df, model_type="random_forest", save=True)
    with st.spinner("Training Isolation Forest..."):
        train_anomaly_detector(df, model_type="isolation_forest", save=True)
    st.success("Models ready!")


# ── Core analysis ─────────────────────────────────────────────────────────

def analyse_signal(signal: np.ndarray, models: dict) -> dict:
    """
    Run full DSP + feature extraction + ML on a signal array.

    Args:
        signal: 1D numpy array from signal input
        models: loaded model dict

    Returns:
        results dict
    """
    # DSP
    freqs, magnitude, phase       = compute_fft(signal)
    fft_data                      = fft_summary(signal)
    spec_freqs, spec_times, Sxx_db = compute_spectrogram(signal)

    # Features
    features    = extract_features(signal)
    feat_vector = extract_features_vector(signal)

    # ML — classification
    fault_name  = "unknown"
    confidence  = 0.0
    class_probs = {}
    is_anomaly  = False
    anomaly_score = None

    if models.get("classifier") is not None:
        from src.ml.train_classifier import predict_fault
        clf = predict_fault(
            feat_vector,
            model=models["classifier"],
            scaler=models["clf_scaler"],
        )
        fault_name  = clf["fault_name"]
        confidence  = clf["confidence"] or 0.0
        class_probs = clf["class_probabilities"]

    if models.get("anomaly") is not None:
        from src.ml.anomaly_detector import detect_anomaly
        ano           = detect_anomaly(
            feat_vector,
            model=models["anomaly"],
            scaler=models["ano_scaler"],
        )
        is_anomaly    = ano["is_anomaly"]
        anomaly_score = ano["anomaly_score"]

    return {
        "signal":       signal,
        "freqs":        freqs,
        "magnitude":    magnitude,
        "fft_data":     fft_data,
        "spec_freqs":   spec_freqs,
        "spec_times":   spec_times,
        "Sxx_db":       Sxx_db,
        "features":     features,
        "feat_vector":  feat_vector,
        "fault_name":   fault_name,
        "confidence":   confidence,
        "class_probs":  class_probs,
        "is_anomaly":   is_anomaly,
        "anomaly_score": anomaly_score,
    }


def run_genai(results: dict) -> dict:
    """Run Groq LLM diagnosis and return diagnosis dict."""
    from src.genai.explainer import run_full_diagnosis
    from src.genai.recommender import get_corrective_actions, build_diagnostic_report

    diagnosis = run_full_diagnosis(
        fault_name=results["fault_name"],
        confidence=results["confidence"],
        features=results["features"],
        fft_data=results["fft_data"],
        anomaly_score=results["anomaly_score"],
        is_anomaly=results["is_anomaly"],
    )
    actions = get_corrective_actions(
        fault_name=diagnosis["fault_name"],
        severity=diagnosis["severity"],
        features=results["features"],
        fault_explanation=diagnosis["explanation"],
        use_llm=True,
    )
    report = build_diagnostic_report(
        fault_name=diagnosis["fault_name"],
        severity=diagnosis["severity"],
        confidence=diagnosis["confidence"],
        explanation=diagnosis["explanation"],
        actions=actions,
        features=results["features"],
        fft_data=results["fft_data"],
    )
    return {**diagnosis, "actions": actions, "report": report}


# ── Analysis page ─────────────────────────────────────────────────────────

def render_analysis_page(options: dict, models: dict):
    """Main analysis page — signal input + full pipeline output."""

    st.title("⚡ Smart Grid Signal Intelligence")
    st.caption("Upload your power grid signal or generate a synthetic one for analysis.")

    # ── Signal input ──────────────────────────────────────────────────────
    signal_data = render_signal_input()

    if signal_data is None:
        st.info("Provide a signal above to begin analysis.")
        return

    signal = signal_data["signal"]

    # ── Analyse button ────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔬 Run Full Analysis", type="primary", use_container_width=True):
        with st.spinner("Running DSP + ML pipeline..."):
            st.session_state["results"]   = analyse_signal(signal, models)
            st.session_state["diagnosis"] = None   # reset on new signal

    results = st.session_state.get("results")
    if not results:
        return

    # ── Fault card + metrics ──────────────────────────────────────────────
    severity = st.session_state.get("severity", "LOW")

    render_fault_card(
        fault_name=results["fault_name"],
        confidence=results["confidence"],
        is_anomaly=results["is_anomaly"],
        severity=severity,
    )
    render_signal_metrics(results["features"], results["fft_data"])

    # ── Class probability bar ─────────────────────────────────────────────
    if results["class_probs"]:
        with st.expander("Class probability breakdown", expanded=False):
            import plotly.graph_objects as go
            probs  = results["class_probs"]
            labels = [k.replace("_", " ").title() for k in probs]
            values = list(probs.values())
            colours = ["#378ADD" if v == max(values) else "#555" for v in values]
            fig = go.Figure(go.Bar(
                x=values, y=labels,
                orientation="h",
                marker_color=colours,
                text=[f"{v:.1%}" for v in values],
                textposition="outside",
            ))
            fig.update_layout(
                height=250,
                margin=dict(l=10, r=60, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(range=[0, 1.1], showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Charts ────────────────────────────────────────────────────────────
    st.markdown("---")
    col_w, col_f = st.columns(2)
    with col_w:
        st.plotly_chart(
            plot_waveform(results["signal"], results["fault_name"],
                          title=f"Waveform — {signal_data['source']}"),
            use_container_width=True
        )
    with col_f:
        st.plotly_chart(
            plot_fft_spectrum(results["freqs"], results["magnitude"],
                              results["fault_name"]),
            use_container_width=True
        )

    col_s, col_r = st.columns(2)
    with col_s:
        if options["show_spectrogram"]:
            st.plotly_chart(
                plot_spectrogram(results["spec_freqs"], results["spec_times"],
                                 results["Sxx_db"]),
                use_container_width=True
            )
    with col_r:
        if options["show_radar"]:
            st.plotly_chart(
                plot_feature_radar(results["features"]),
                use_container_width=True
            )

    if options["show_harmonics"]:
        st.plotly_chart(
            plot_harmonic_bars(results["features"]),
            use_container_width=True
        )

    # ── AI Diagnostics ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🤖 AI Diagnostics")

    if not options["run_genai"]:
        st.info("Enable 'AI diagnostics' in the sidebar to get Groq-powered explanations.")
        return

    if st.button("💬 Run AI Diagnosis", type="secondary"):
        with st.spinner("Consulting Groq LLM..."):
            try:
                diag = run_genai(results)
                st.session_state["diagnosis"] = diag
                st.session_state["severity"]  = diag["severity"]
            except Exception as e:
                st.error(f"GenAI error: {e}")
                return

    diagnosis = st.session_state.get("diagnosis")
    if not diagnosis:
        return

    render_header(
        summary=diagnosis.get("summary", ""),
        severity=diagnosis.get("severity", "LOW"),
    )

    col_e, col_a = st.columns([3, 2])
    with col_e:
        render_explanation(
            explanation=diagnosis.get("explanation", ""),
            reason=diagnosis.get("reason", ""),
        )
    with col_a:
        render_corrective_actions(diagnosis.get("actions", []))

    render_report(diagnosis.get("report", ""))


# ── Training page ─────────────────────────────────────────────────────────

def render_training_page():
    """Model training page."""
    st.title("🏋️ Model Training")
    st.markdown("Train all models from scratch using synthetic signal data.")

    col1, col2 = st.columns(2)
    with col1:
        n_per_class = st.slider("Samples per fault class", 100, 500, 200, 50)
    with col2:
        train_lstm = st.checkbox("Train LSTM (slower, ~5 min)", value=False)

    if st.button("🚀 Train all models", type="primary"):
        from src.simulation.signal_generator import generate_dataset
        from src.features.feature_pipeline import build_feature_dataframe, save_features
        from src.ml.train_classifier import train_classifier
        from src.ml.anomaly_detector import train_anomaly_detector

        with st.spinner("Generating dataset..."):
            X, y = generate_dataset(n_per_class=n_per_class, add_noise=True)
            st.success(f"Generated {len(X)} signals")

        with st.spinner("Extracting features..."):
            df = build_feature_dataframe(X, y)
            save_features(df)
            st.success(f"Feature matrix: {df.shape}")

        with st.spinner("Training Random Forest..."):
            clf = train_classifier(df, save=True)
            st.success(f"Random Forest — Accuracy: {clf['metrics']['accuracy']:.4f}  "
                       f"F1: {clf['metrics']['f1_macro']:.4f}")

        with st.spinner("Training Isolation Forest..."):
            ano = train_anomaly_detector(df, save=True)
            st.success(f"Isolation Forest — Anomaly F1: {ano['metrics']['anomaly_f1']:.4f}")

        if train_lstm:
            from src.ml.lstm_predictor import train_lstm_classifier
            with st.spinner("Training LSTM..."):
                lstm = train_lstm_classifier(X, y, save=True)
                st.success(f"LSTM — Accuracy: {lstm['metrics']['accuracy']:.4f}")

        st.balloons()
        st.success("All models saved. Reload the app to use them.")
        st.cache_resource.clear()


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    models  = load_models()
    options = render_sidebar()

    if options["page"] == "Train Models":
        render_training_page()
    else:
        render_analysis_page(options, models)


if __name__ == "__main__":
    main()
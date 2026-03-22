"""
Layer 6 — Dashboard
plots.py

All Plotly chart builders for the Streamlit dashboard.
Each function takes processed data and returns a Plotly figure.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from config import SAMPLING_RATE, SIGNAL_FREQ, FAULT_TYPES


# ── Colour palette ────────────────────────────────────────────────────────

FAULT_COLOURS = {
    "normal":               "#2ECC71",
    "harmonic_distortion":  "#E74C3C",
    "voltage_sag":          "#E67E22",
    "voltage_swell":        "#9B59B6",
    "transient_spike":      "#E91E63",
    "frequency_deviation":  "#3498DB",
}

SEVERITY_COLOURS = {
    "LOW":      "#2ECC71",
    "MEDIUM":   "#F39C12",
    "HIGH":     "#E74C3C",
    "CRITICAL": "#8E1A0E",
}


def _base_layout(fig: go.Figure, title: str = "", height: int = 350) -> go.Figure:
    """Apply consistent dark-friendly layout to any figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=height,
        margin=dict(l=50, r=20, t=45, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
    )
    return fig


# ── Signal waveform ───────────────────────────────────────────────────────

def plot_waveform(
    signal: np.ndarray,
    fault_name: str = "normal",
    sampling_rate: int = SAMPLING_RATE,
    title: str = "Signal waveform",
    max_points: int = 2000,
) -> go.Figure:
    """
    Time-domain waveform plot.

    Args:
        signal:       raw signal array
        fault_name:   used for line colour
        sampling_rate: samples per second
        max_points:   downsample for performance

    Returns:
        Plotly Figure
    """
    n = len(signal)
    t = np.linspace(0, n / sampling_rate, n)

    # Downsample for rendering speed
    if n > max_points:
        step   = n // max_points
        signal = signal[::step]
        t      = t[::step]

    colour = FAULT_COLOURS.get(fault_name, "#378ADD")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=signal,
        mode="lines",
        line=dict(color=colour, width=1.2),
        name=fault_name.replace("_", " ").title(),
    ))

    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Amplitude (p.u.)")
    return _base_layout(fig, title)


def plot_waveform_comparison(
    signals: dict,
    sampling_rate: int = SAMPLING_RATE,
    max_points: int = 1000,
) -> go.Figure:
    """
    Overlay multiple fault waveforms on the same plot.

    Args:
        signals: dict of {fault_name: signal_array}
    """
    fig = go.Figure()

    for fault_name, signal in signals.items():
        n = len(signal)
        t = np.linspace(0, n / sampling_rate, n)
        if n > max_points:
            step   = n // max_points
            signal = signal[::step]
            t      = t[::step]

        fig.add_trace(go.Scatter(
            x=t, y=signal,
            mode="lines",
            line=dict(color=FAULT_COLOURS.get(fault_name, "#888"), width=1.2),
            name=fault_name.replace("_", " ").title(),
        ))

    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Amplitude (p.u.)")
    return _base_layout(fig, "Waveform comparison — all fault types", height=400)


# ── FFT spectrum ──────────────────────────────────────────────────────────

def plot_fft_spectrum(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    fault_name: str = "normal",
    max_freq: float = 1000.0,
    title: str = "FFT magnitude spectrum",
) -> go.Figure:
    """
    Single-sided FFT magnitude spectrum with harmonic markers.

    Args:
        freqs:     frequency axis (Hz)
        magnitude: amplitude spectrum
        fault_name: used for colour
        max_freq:  upper frequency limit for display
    """
    mask = freqs <= max_freq
    f    = freqs[mask]
    m    = magnitude[mask]

    colour = FAULT_COLOURS.get(fault_name, "#378ADD")

    # Convert hex to rgba for Plotly fill colour
    def hex_to_rgba(hex_colour: str, alpha: float = 0.15) -> str:
        h = hex_colour.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fill_colour = hex_to_rgba(colour, alpha=0.15)

    fig = go.Figure()

    # Spectrum line
    fig.add_trace(go.Scatter(
        x=f, y=m,
        mode="lines",
        fill="tozeroy",
        fillcolor=fill_colour,
        line=dict(color=colour, width=1.5),
        name="Spectrum",
    ))

    # Harmonic markers (50, 150, 250, 350, 450, 550 Hz)
    harmonics = [SIGNAL_FREQ * h for h in [1, 3, 5, 7, 9, 11] if SIGNAL_FREQ * h <= max_freq]
    for h_freq in harmonics:
        fig.add_vline(
            x=h_freq,
            line=dict(color="rgba(255,200,0,0.4)", width=1, dash="dot"),
            annotation_text=f"{int(h_freq)}Hz",
            annotation_font_size=9,
        )

    fig.update_xaxes(title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="Amplitude (p.u.)")
    return _base_layout(fig, title)


# ── Spectrogram ───────────────────────────────────────────────────────────

def plot_spectrogram(
    freqs: np.ndarray,
    times: np.ndarray,
    Sxx_db: np.ndarray,
    max_freq: float = 1000.0,
    title: str = "Spectrogram (STFT)",
) -> go.Figure:
    """
    Heatmap-style spectrogram showing time-frequency energy distribution.

    Args:
        freqs:   frequency axis (Hz)
        times:   time axis (s)
        Sxx_db:  spectrogram matrix in dB
        max_freq: upper frequency to display
    """
    mask  = freqs <= max_freq
    f     = freqs[mask]
    S_trunc = Sxx_db[mask, :]

    fig = go.Figure(go.Heatmap(
        x=times,
        y=f,
        z=S_trunc,
        colorscale="Viridis",
        colorbar=dict(title="dB", thickness=12),
        zsmooth="best",
    ))

    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Frequency (Hz)")
    return _base_layout(fig, title, height=320)


# ── Feature radar chart ───────────────────────────────────────────────────

def plot_feature_radar(
    features: dict,
    title: str = "Signal feature profile",
) -> go.Figure:
    """
    Radar/spider chart of normalised key features.
    Gives a quick visual fingerprint of the signal's health.
    """
    # Select and normalise key features for display
    display_features = {
        "THD %":             min(features.get("thd_percent", 0) / 25.0, 1.0),
        "Crest factor":      min((features.get("crest_factor", 1.414) - 1.0) / 3.0, 1.0),
        "Spectral entropy":  min(features.get("spectral_entropy", 0) / 5.0, 1.0),
        "Freq deviation":    min(abs(features.get("freq_deviation_hz", 0)) / 5.0, 1.0),
        "Waveform dev":      min(features.get("waveform_deviation", 0) / 1.0, 1.0),
        "Kurtosis":          min((features.get("kurtosis", 3) - 3.0) / 10.0, 1.0),
        "Harmonic ratio":    min(features.get("harmonic_energy_ratio", 0) / 0.5, 1.0),
    }

    cats   = list(display_features.keys())
    values = list(display_features.values())
    # Close the polygon
    cats.append(cats[0])
    values.append(values[0])

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=cats,
        fill="toself",
        fillcolor="rgba(55, 138, 221, 0.2)",
        line=dict(color="#378ADD", width=2),
        name="Signal",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        height=350,
        margin=dict(l=60, r=60, t=45, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, font=dict(size=14)),
    )

    return fig


# ── Harmonic bar chart ────────────────────────────────────────────────────

def plot_harmonic_bars(
    features: dict,
    title: str = "Individual harmonic distortion (%)",
) -> go.Figure:
    """
    Bar chart of IHD per harmonic order with IEEE 519 limit line.
    """
    orders = [3, 5, 7, 9, 11]
    freqs  = [f"{SIGNAL_FREQ * o:.0f} Hz (H{o})" for o in orders]
    values = [features.get(f"ihd_{o}", 0.0) for o in orders]
    colours = [
        "#E74C3C" if v > 5.0 else "#F39C12" if v > 2.0 else "#2ECC71"
        for v in values
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=freqs, y=values,
        marker_color=colours,
        name="IHD %",
        text=[f"{v:.2f}%" for v in values],
        textposition="outside",
    ))

    # IEEE 519 individual harmonic limit
    fig.add_hline(
        y=5.0,
        line=dict(color="#E74C3C", width=1.5, dash="dash"),
        annotation_text="IEEE 519 limit (5%)",
        annotation_font_size=10,
    )

    fig.update_xaxes(title_text="Harmonic frequency")
    fig.update_yaxes(title_text="IHD (%)", rangemode="tozero")
    return _base_layout(fig, title, height=320)


# ── Confusion matrix ──────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: list,
    class_names: list = None,
    title: str = "Confusion matrix",
) -> go.Figure:
    """
    Annotated heatmap confusion matrix.

    Args:
        cm:           confusion matrix as list of lists
        class_names:  class label strings
    """
    if class_names is None:
        class_names = [FAULT_TYPES[i] for i in sorted(FAULT_TYPES.keys())]

    short_names = [n.replace("_", " ").title() for n in class_names]
    cm_arr = np.array(cm)

    # Normalise for colour scale
    cm_norm = cm_arr / (cm_arr.sum(axis=1, keepdims=True) + 1e-12)

    annotations = [[str(cm_arr[i][j]) for j in range(len(cm_arr[i]))]
                   for i in range(len(cm_arr))]

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=short_names,
        y=short_names,
        colorscale="Blues",
        showscale=True,
        colorbar=dict(title="Recall", thickness=12),
        text=annotations,
        texttemplate="%{text}",
        textfont=dict(size=11),
    ))

    fig.update_xaxes(title_text="Predicted", tickangle=30)
    fig.update_yaxes(title_text="Actual", autorange="reversed")
    return _base_layout(fig, title, height=400)


# ── Feature importance ────────────────────────────────────────────────────

def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Feature importance (Random Forest)",
) -> go.Figure:
    """
    Horizontal bar chart of top-N feature importances.
    """
    df = importance_df.head(top_n).sort_values("importance")

    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker_color="#378ADD",
        text=[f"{v:.4f}" for v in df["importance"]],
        textposition="outside",
    ))

    fig.update_xaxes(title_text="Importance")
    fig.update_yaxes(title_text="")
    return _base_layout(fig, title, height=max(300, top_n * 22))


# ── LSTM training history ─────────────────────────────────────────────────

def plot_training_history(
    history: dict,
    title: str = "LSTM training history",
) -> go.Figure:
    """
    Dual-axis plot of training/validation loss and accuracy over epochs.
    """
    epochs = list(range(1, len(history.get("loss", [])) + 1))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Loss", "Accuracy"),
    )

    # Loss
    fig.add_trace(go.Scatter(
        x=epochs, y=history.get("loss", []),
        mode="lines", name="Train loss",
        line=dict(color="#378ADD", width=2),
    ), row=1, col=1)

    if "val_loss" in history:
        fig.add_trace(go.Scatter(
            x=epochs, y=history["val_loss"],
            mode="lines", name="Val loss",
            line=dict(color="#E74C3C", width=2, dash="dash"),
        ), row=1, col=1)

    # Accuracy
    if "accuracy" in history:
        fig.add_trace(go.Scatter(
            x=epochs, y=history["accuracy"],
            mode="lines", name="Train acc",
            line=dict(color="#2ECC71", width=2),
        ), row=1, col=2)

    if "val_accuracy" in history:
        fig.add_trace(go.Scatter(
            x=epochs, y=history["val_accuracy"],
            mode="lines", name="Val acc",
            line=dict(color="#F39C12", width=2, dash="dash"),
        ), row=1, col=2)

    fig.update_layout(
        height=320,
        margin=dict(l=50, r=20, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, font=dict(size=14)),
    )
    return fig


# ── Anomaly score timeline ────────────────────────────────────────────────

def plot_anomaly_scores(
    scores: np.ndarray,
    y_true: np.ndarray = None,
    threshold: float = None,
    title: str = "Anomaly scores (Isolation Forest)",
) -> go.Figure:
    """
    Scatter plot of anomaly scores with optional threshold line and
    colour-coded points by true label.
    """
    n      = len(scores)
    index  = np.arange(n)
    colours = ["#E74C3C" if (y_true is not None and y_true[i] > 0) else "#2ECC71"
               for i in range(n)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=index, y=scores,
        mode="markers",
        marker=dict(color=colours, size=4, opacity=0.6),
        name="Anomaly score",
    ))

    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line=dict(color="#F39C12", width=1.5, dash="dash"),
            annotation_text=f"Threshold ({threshold:.4f})",
            annotation_font_size=10,
        )

    fig.update_xaxes(title_text="Sample index")
    fig.update_yaxes(title_text="Score (higher = more normal)")
    return _base_layout(fig, title, height=300)
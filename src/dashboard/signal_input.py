"""
Layer 6 — Dashboard
signal_input.py

Handles all user signal input methods:
  1. File upload   — CSV, NPY, TXT from real instruments
  2. Synthetic     — generate from fault type selector
  3. Manual entry  — paste raw comma-separated values

Validates, resamples, and normalises the signal before
handing it to the analysis pipeline.
"""

import numpy as np
import pandas as pd
import streamlit as st
from config import SAMPLING_RATE, NUM_SAMPLES, SIGNAL_FREQ, FAULT_TYPES


# ── Validation ────────────────────────────────────────────────────────────

MIN_SAMPLES = 1000       # at least 0.1 seconds at 10kHz
MAX_SAMPLES = 200_000    # hard cap to prevent memory issues


def validate_signal(signal: np.ndarray) -> tuple:
    """
    Validate a user-supplied signal array.

    Returns:
        (is_valid: bool, message: str)
    """
    if signal is None or len(signal) == 0:
        return False, "Signal is empty."

    if len(signal) < MIN_SAMPLES:
        return False, (
            f"Signal too short: {len(signal)} samples. "
            f"Minimum is {MIN_SAMPLES} samples "
            f"({MIN_SAMPLES/SAMPLING_RATE:.2f}s at {SAMPLING_RATE} Hz)."
        )

    if len(signal) > MAX_SAMPLES:
        return False, (
            f"Signal too long: {len(signal)} samples. "
            f"Maximum is {MAX_SAMPLES} samples. Please trim your data."
        )

    if not np.isfinite(signal).all():
        n_bad = (~np.isfinite(signal)).sum()
        return False, f"Signal contains {n_bad} NaN or Inf values. Please clean your data."

    if np.std(signal) < 1e-10:
        return False, "Signal appears to be all zeros or constant. No analysis possible."

    return True, "OK"


def preprocess_signal(
    signal: np.ndarray,
    target_samples: int = NUM_SAMPLES,
    normalise: bool = False,
) -> np.ndarray:
    """
    Prepare a raw user signal for the analysis pipeline.

    Steps:
      1. Truncate or zero-pad to target_samples
      2. Optionally normalise to [-1, 1]

    Args:
        signal:         raw 1D signal array
        target_samples: output length (default NUM_SAMPLES = 10,000)
        normalise:      scale to [-1, 1] peak

    Returns:
        processed signal of shape (target_samples,)
    """
    signal = signal.astype(np.float32)

    if len(signal) >= target_samples:
        # Take the first target_samples
        signal = signal[:target_samples]
    else:
        # Zero-pad at the end
        pad = target_samples - len(signal)
        signal = np.pad(signal, (0, pad), mode="constant")

    if normalise:
        peak = np.max(np.abs(signal))
        if peak > 1e-10:
            signal = signal / peak

    return signal


# ── Parsers ───────────────────────────────────────────────────────────────

def parse_csv(uploaded_file, column: str = None) -> tuple:
    """
    Parse a CSV file into a 1D signal array.

    Supports:
      - Single-column CSV  (just values, no header)
      - Multi-column CSV   (user picks which column)
      - Time + value CSV   (two columns: time, voltage)

    Returns:
        (signal: np.ndarray, detected_sampling_rate: float, column_names: list)
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {e}")

    col_names = list(df.columns)

    # Try to detect numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found in CSV.")

    # If column specified, use it
    if column and column in df.columns:
        signal = df[column].dropna().values.astype(np.float32)
        return signal, SAMPLING_RATE, col_names

    # If two columns: assume [time, value]
    if len(numeric_cols) == 2:
        time_col  = numeric_cols[0]
        value_col = numeric_cols[1]

        times  = df[time_col].dropna().values.astype(np.float64)
        signal = df[value_col].dropna().values.astype(np.float32)

        # Estimate sampling rate from time column
        if len(times) > 1:
            dt = np.median(np.diff(times))
            detected_fs = round(1.0 / dt) if dt > 0 else SAMPLING_RATE
        else:
            detected_fs = SAMPLING_RATE

        return signal, detected_fs, col_names

    # Single numeric column or pick first
    signal = df[numeric_cols[0]].dropna().values.astype(np.float32)
    return signal, SAMPLING_RATE, col_names


def parse_npy(uploaded_file) -> np.ndarray:
    """Parse a .npy file into a 1D signal array."""
    try:
        arr = np.load(uploaded_file, allow_pickle=False)
    except Exception as e:
        raise ValueError(f"Could not load .npy file: {e}")

    if arr.ndim > 1:
        # Take first row or flatten
        arr = arr.flatten()[:NUM_SAMPLES * 2]

    return arr.astype(np.float32)


def parse_txt(uploaded_file) -> np.ndarray:
    """
    Parse a plain text file — one value per line or space/comma separated.
    Common output format from oscilloscopes and data loggers.
    """
    try:
        content = uploaded_file.read().decode("utf-8")
        # Try comma-separated first, then whitespace
        if "," in content:
            values = [float(v.strip()) for v in content.split(",") if v.strip()]
        else:
            values = [float(v.strip()) for v in content.split() if v.strip()]
        return np.array(values, dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Could not parse text file: {e}")


# ── Main input widget ─────────────────────────────────────────────────────

def render_signal_input() -> dict:
    """
    Full signal input UI rendered in the main area (not sidebar).
    Returns a dict with the signal and metadata, or None if no signal yet.

    Returns:
        dict with keys:
            signal, source, sampling_rate, label, n_samples, duration_s
        or None if user hasn't provided a signal yet.
    """
    st.markdown("## Signal Input")

    input_method = st.radio(
        "Choose input method",
        ["Upload file", "Generate synthetic", "Paste values"],
        horizontal=True,
        label_visibility="collapsed",
    )

    signal      = None
    source      = None
    fs          = SAMPLING_RATE
    label       = "user_signal"
    extra_info  = {}

    # ── Method 1: File upload ─────────────────────────────────────────────
    if input_method == "Upload file":
        st.markdown("#### Upload your signal file")

        col_info, col_upload = st.columns([2, 1])

        with col_info:
            st.markdown("""
            **Supported formats:**
            - **CSV** — one column of voltage values, or two columns (time, voltage)
            - **NPY** — NumPy array saved with `np.save()`
            - **TXT** — one value per line or comma-separated

            **Expected signal:**
            - 50 Hz power grid voltage (single phase)
            - Sampling rate: ideally 10 kHz (10,000 samples/sec)
            - Duration: at least 0.1 seconds

            **Example CSV format:**
            ```
            voltage
            0.9823
            0.9901
            1.0023
            ...
            ```
            """)

        with col_upload:
            uploaded = st.file_uploader(
                "Drop your signal file here",
                type=["csv", "npy", "txt"],
                label_visibility="collapsed",
            )

            if uploaded:
                st.caption(f"File: `{uploaded.name}` ({uploaded.size / 1024:.1f} KB)")

        if uploaded:
            try:
                ext = uploaded.name.lower().split(".")[-1]

                if ext == "csv":
                    # First pass to detect columns
                    import io
                    raw_bytes = uploaded.read()
                    uploaded.seek(0)

                    preview_df = pd.read_csv(io.BytesIO(raw_bytes))
                    numeric_cols = preview_df.select_dtypes(include=[np.number]).columns.tolist()

                    selected_col = None
                    if len(numeric_cols) > 2:
                        selected_col = st.selectbox(
                            "Select the voltage/signal column:",
                            options=numeric_cols,
                        )

                    signal, fs, col_names = parse_csv(uploaded, column=selected_col)
                    source = f"CSV ({uploaded.name})"
                    extra_info["columns"] = col_names

                elif ext == "npy":
                    signal = parse_npy(uploaded)
                    source = f"NPY ({uploaded.name})"

                elif ext == "txt":
                    signal = parse_txt(uploaded)
                    source = f"TXT ({uploaded.name})"

                # Let user confirm or override sampling rate
                st.markdown("---")
                col_fs, col_norm = st.columns(2)
                with col_fs:
                    fs = st.number_input(
                        "Sampling rate (Hz)",
                        min_value=1000, max_value=100000,
                        value=int(fs), step=1000,
                        help="Samples per second of your recording device"
                    )
                with col_norm:
                    normalise = st.checkbox(
                        "Normalise signal to [-1, 1]",
                        value=False,
                        help="Enable if your signal has very large or unusual amplitude"
                    )

                signal = preprocess_signal(signal, normalise=normalise)

            except ValueError as e:
                st.error(f"File error: {e}")
                signal = None

    # ── Method 2: Synthetic ───────────────────────────────────────────────
    elif input_method == "Generate synthetic":
        st.markdown("#### Generate a synthetic power grid signal")
        st.caption("Useful for testing the system or demonstrating fault types.")

        col1, col2, col3 = st.columns(3)

        with col1:
            fault_names = [FAULT_TYPES[i] for i in sorted(FAULT_TYPES.keys())]
            selected_fault = st.selectbox(
                "Fault type",
                options=fault_names,
                format_func=lambda x: x.replace("_", " ").title(),
            )
            fault_type = [k for k, v in FAULT_TYPES.items() if v == selected_fault][0]

        with col2:
            add_noise = st.checkbox("Add realistic noise", value=True)
            noise_snr = st.slider(
                "SNR (dB)", min_value=10, max_value=50, value=30, step=5,
                disabled=not add_noise,
            )

        with col3:
            normalise = st.checkbox("Normalise to [-1, 1]", value=False)

        if st.button("Generate signal", type="primary"):
            from src.simulation.signal_generator import generate_signal
            from src.simulation.noise_utils import add_combined_noise

            sig = generate_signal(fault_type=fault_type)
            if add_noise:
                sig = add_combined_noise(sig, snr_db=noise_snr)
            signal = preprocess_signal(sig, normalise=normalise)
            source = f"Synthetic — {selected_fault}"
            label  = selected_fault
            st.session_state["last_signal"]  = signal
            st.session_state["last_source"]  = source
            st.session_state["last_label"]   = label

        # Use last generated signal if available
        if signal is None and "last_signal" in st.session_state:
            signal = st.session_state["last_signal"]
            source = st.session_state.get("last_source", "Synthetic")
            label  = st.session_state.get("last_label", "unknown")

    # ── Method 3: Paste values ────────────────────────────────────────────
    elif input_method == "Paste values":
        st.markdown("#### Paste raw signal values")

        col_ex, col_input = st.columns([1, 2])

        with col_ex:
            st.markdown("""
            Paste comma-separated or newline-separated voltage samples.

            **Example (first 5 values of a 50Hz sine):**
            ```
            0.000, 0.031, 0.063, 0.094, 0.125
            ```
            Minimum **1,000 values** required.
            """)

            fs = st.number_input(
                "Sampling rate (Hz)",
                min_value=1000, max_value=100000,
                value=10000, step=1000,
            )
            normalise = st.checkbox("Normalise to [-1, 1]", value=False)

        with col_input:
            raw_text = st.text_area(
                "Paste signal values here:",
                height=280,
                placeholder="0.000, 0.031, 0.063, 0.094 ...",
                label_visibility="collapsed",
            )

        if raw_text.strip():
            try:
                # Support both comma and newline separated
                raw_text = raw_text.replace("\n", ",").replace(";", ",")
                values   = [float(v.strip()) for v in raw_text.split(",")
                            if v.strip()]
                signal   = np.array(values, dtype=np.float32)
                signal   = preprocess_signal(signal, normalise=normalise)
                source   = "Manual entry"
            except ValueError as e:
                st.error(f"Could not parse values: {e}. Make sure all values are numbers.")
                signal = None

    # ── Validation & preview ──────────────────────────────────────────────
    if signal is not None:
        is_valid, msg = validate_signal(signal)

        if not is_valid:
            st.error(f"Signal validation failed: {msg}")
            return None

        n_samples  = len(signal)
        duration_s = n_samples / fs

        # Signal preview
        st.markdown("---")
        st.markdown("#### Signal preview")

        col_stats, col_chart = st.columns([1, 3])

        with col_stats:
            st.metric("Samples",    f"{n_samples:,}")
            st.metric("Duration",   f"{duration_s:.3f} s")
            st.metric("Sample rate", f"{fs:,} Hz")
            st.metric("Peak",       f"{np.max(np.abs(signal)):.4f}")
            st.metric("RMS",        f"{np.sqrt(np.mean(signal**2)):.4f}")
            if source:
                st.caption(f"Source: {source}")

        with col_chart:
            import plotly.graph_objects as go
            t       = np.linspace(0, duration_s, n_samples)
            step    = max(1, n_samples // 2000)
            fig     = go.Figure()
            fig.add_trace(go.Scatter(
                x=t[::step], y=signal[::step],
                mode="lines",
                line=dict(color="#378ADD", width=1.2),
                name="Signal",
            ))
            fig.update_layout(
                height=220,
                margin=dict(l=40, r=10, t=10, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Time (s)",
                           showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
                yaxis=dict(title="Amplitude",
                           showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
            )
            st.plotly_chart(fig, use_container_width=True)

        return {
            "signal":       signal,
            "source":       source or "unknown",
            "sampling_rate": int(fs),
            "label":        label,
            "n_samples":    n_samples,
            "duration_s":   round(duration_s, 4),
        }

    return None
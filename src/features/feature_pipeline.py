"""
Layer 3 — Feature Extraction
feature_pipeline.py

Assembles all time-domain and frequency-domain features into a
single flat feature vector per signal window.
Produces the Pandas DataFrame that feeds into ML training.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from config import (
    SAMPLING_RATE, SIGNAL_FREQ, FAULT_TYPES,
    WINDOW_SIZE, WINDOW_STEP, DATA_PROCESSED
)
from src.features.thd_calculator import (
    compute_thd, compute_individual_harmonic_distortion,
    compute_power_factor_distortion
)
from src.features.spectral_features import (
    compute_rms, compute_peak_value, compute_crest_factor,
    compute_form_factor, compute_kurtosis, compute_skewness,
    compute_zero_crossing_rate, compute_signal_energy,
    compute_variance, compute_spectral_entropy,
    compute_spectral_centroid, compute_spectral_bandwidth,
    compute_spectral_flatness, compute_harmonic_energy_ratio,
    compute_fundamental_amplitude, compute_peak_count,
    compute_voltage_unbalance, compute_waveform_deviation
)
from src.dsp.fft_analyzer import detect_frequency_deviation


# ── Feature names (defines column order) ─────────────────────────────────

FEATURE_NAMES = [
    # time-domain
    "rms",
    "peak_value",
    "crest_factor",
    "form_factor",
    "kurtosis",
    "skewness",
    "zero_crossing_rate",
    "signal_energy",
    "variance",
    "voltage_unbalance",
    "waveform_deviation",
    "peak_count",
    # frequency-domain
    "spectral_entropy",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_flatness",
    "harmonic_energy_ratio",
    "fundamental_amplitude",
    "thd_percent",
    "distortion_power_factor",
    "freq_deviation_hz",
    # individual harmonic distortions
    "ihd_3",
    "ihd_5",
    "ihd_7",
    "ihd_9",
    "ihd_11",
]


def extract_features(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE
) -> dict:
    """
    Extract all features from a single signal window.

    Args:
        signal:        input waveform (any length)
        sampling_rate: samples per second

    Returns:
        dict mapping feature_name → scalar value
    """
    ihd = compute_individual_harmonic_distortion(signal, SIGNAL_FREQ, sampling_rate)
    thd = compute_thd(signal, SIGNAL_FREQ, sampling_rate)
    freq_dev = detect_frequency_deviation(signal, SIGNAL_FREQ, sampling_rate)

    features = {
        # time-domain
        "rms":                    compute_rms(signal),
        "peak_value":             compute_peak_value(signal),
        "crest_factor":           compute_crest_factor(signal),
        "form_factor":            compute_form_factor(signal),
        "kurtosis":               compute_kurtosis(signal),
        "skewness":               compute_skewness(signal),
        "zero_crossing_rate":     compute_zero_crossing_rate(signal),
        "signal_energy":          compute_signal_energy(signal),
        "variance":               compute_variance(signal),
        "voltage_unbalance":      compute_voltage_unbalance(signal),
        "waveform_deviation":     compute_waveform_deviation(signal, sampling_rate),
        "peak_count":             compute_peak_count(signal),
        # frequency-domain
        "spectral_entropy":       compute_spectral_entropy(signal, sampling_rate),
        "spectral_centroid":      compute_spectral_centroid(signal, sampling_rate),
        "spectral_bandwidth":     compute_spectral_bandwidth(signal, sampling_rate),
        "spectral_flatness":      compute_spectral_flatness(signal, sampling_rate),
        "harmonic_energy_ratio":  compute_harmonic_energy_ratio(signal, SIGNAL_FREQ, sampling_rate),
        "fundamental_amplitude":  compute_fundamental_amplitude(signal, SIGNAL_FREQ, sampling_rate),
        "thd_percent":            thd,
        "distortion_power_factor": compute_power_factor_distortion(thd),
        "freq_deviation_hz":      freq_dev["deviation_hz"],
        # individual harmonics
        "ihd_3":                  ihd.get(3, 0.0),
        "ihd_5":                  ihd.get(5, 0.0),
        "ihd_7":                  ihd.get(7, 0.0),
        "ihd_9":                  ihd.get(9, 0.0),
        "ihd_11":                 ihd.get(11, 0.0),
    }

    return features


def extract_features_vector(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE
) -> np.ndarray:
    """
    Extract features and return as a numpy array (for ML inference).
    Order follows FEATURE_NAMES list.

    Returns:
        np.ndarray of shape (n_features,)
    """
    feat_dict = extract_features(signal, sampling_rate)
    return np.array([feat_dict[k] for k in FEATURE_NAMES], dtype=np.float32)


def build_feature_dataframe(
    X: np.ndarray,
    y: np.ndarray,
    sampling_rate: int = SAMPLING_RATE
) -> pd.DataFrame:
    """
    Build a Pandas DataFrame of features from a signal dataset.

    Args:
        X: signal array of shape (n_samples, n_signal_points)
        y: label array of shape (n_samples,)
        sampling_rate: samples per second

    Returns:
        DataFrame with columns = FEATURE_NAMES + ['label', 'fault_name']
    """
    rows = []
    n = len(X)

    print(f"Extracting features from {n} signals...")
    for i, (signal, label) in enumerate(zip(X, y)):
        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"  [{i+1}/{n}]")

        feat = extract_features(signal, sampling_rate)
        feat["label"]      = int(label)
        feat["fault_name"] = FAULT_TYPES.get(int(label), "unknown")
        rows.append(feat)

    df = pd.DataFrame(rows)

    # Enforce column order
    cols = FEATURE_NAMES + ["label", "fault_name"]
    return df[cols]


def windowed_features(
    signal: np.ndarray,
    window_size: int = WINDOW_SIZE,
    step: int = WINDOW_STEP,
    sampling_rate: int = SAMPLING_RATE
) -> pd.DataFrame:
    """
    Extract features from overlapping windows of a long signal.
    Used in real-time / streaming mode in the dashboard.

    Args:
        signal:      long input waveform
        window_size: samples per window
        step:        step size between windows (overlap = window_size - step)
        sampling_rate: samples per second

    Returns:
        DataFrame where each row is one window's features
        Includes 'window_start_s' and 'window_end_s' columns
    """
    rows = []
    starts = range(0, len(signal) - window_size + 1, step)

    for start in starts:
        end    = start + window_size
        window = signal[start:end]

        feat = extract_features(window, sampling_rate)
        feat["window_start_s"] = round(start / sampling_rate, 4)
        feat["window_end_s"]   = round(end   / sampling_rate, 4)
        rows.append(feat)

    cols = ["window_start_s", "window_end_s"] + FEATURE_NAMES
    return pd.DataFrame(rows)[cols]


def save_features(df: pd.DataFrame, filename: str = "features.csv") -> Path:
    """
    Save feature DataFrame to the processed data directory.

    Args:
        df:       feature DataFrame
        filename: output filename

    Returns:
        Path to saved file
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / filename
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows → {out_path}")
    return out_path


def load_features(filename: str = "features.csv") -> pd.DataFrame:
    """
    Load feature DataFrame from the processed data directory.

    Args:
        filename: input filename

    Returns:
        feature DataFrame
    """
    path = DATA_PROCESSED / filename
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    return pd.read_csv(path)
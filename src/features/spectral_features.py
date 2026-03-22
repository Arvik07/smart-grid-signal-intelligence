"""
Layer 3 — Feature Extraction
spectral_features.py

Time-domain and frequency-domain feature extraction for ML classification.
Each function takes a raw signal and returns a scalar feature.
The feature_pipeline.py assembles all of these into a single feature vector.
"""

import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from config import SAMPLING_RATE, SIGNAL_FREQ
from src.dsp.fft_analyzer import compute_fft, get_harmonic_amplitudes


# ── Time-domain features ──────────────────────────────────────────────────

def compute_rms(signal: np.ndarray) -> float:
    """
    Root Mean Square — primary measure of signal energy.
    Normal 50Hz sinusoid of amplitude 1.0 → RMS = 1/sqrt(2) ≈ 0.707
    """
    return float(np.sqrt(np.mean(signal ** 2)))


def compute_peak_value(signal: np.ndarray) -> float:
    """Maximum absolute amplitude of the signal."""
    return float(np.max(np.abs(signal)))


def compute_crest_factor(signal: np.ndarray) -> float:
    """
    Crest Factor = peak / RMS.
    Pure sinusoid → CF = sqrt(2) ≈ 1.414.
    Spikes dramatically increase CF. Clipped signals reduce it.
    """
    rms = compute_rms(signal)
    if rms < 1e-12:
        return 0.0
    return float(compute_peak_value(signal) / rms)


def compute_form_factor(signal: np.ndarray) -> float:
    """
    Form Factor = RMS / mean(|signal|).
    Pure sinusoid → FF = π/(2*sqrt(2)) ≈ 1.1107.
    Sensitive to waveform shape distortion.
    """
    mean_abs = np.mean(np.abs(signal))
    if mean_abs < 1e-12:
        return 0.0
    return float(compute_rms(signal) / mean_abs)


def compute_kurtosis(signal: np.ndarray) -> float:
    """
    Statistical kurtosis — measures tail heaviness.
    Gaussian noise → kurtosis ≈ 3.
    Transient spikes push kurtosis >> 3.
    """
    return float(stats.kurtosis(signal, fisher=False))


def compute_skewness(signal: np.ndarray) -> float:
    """
    Statistical skewness — measures signal asymmetry.
    Symmetric signals (sine, normal) → skewness ≈ 0.
    """
    return float(stats.skew(signal))


def compute_zero_crossing_rate(signal: np.ndarray) -> float:
    """
    Zero-Crossing Rate (ZCR) — number of sign changes per second.
    50 Hz pure sinusoid → ZCR = 100 crossings/sec (2 per cycle).
    Harmonic distortion and frequency deviation shift ZCR.
    """
    crossings = np.where(np.diff(np.sign(signal)))[0]
    duration  = len(signal) / SAMPLING_RATE  # seconds
    return float(len(crossings) / duration)


def compute_signal_energy(signal: np.ndarray) -> float:
    """
    Total signal energy = sum of squared samples.
    Voltage sag reduces energy; swell increases it.
    """
    return float(np.sum(signal ** 2))


def compute_mean_absolute_value(signal: np.ndarray) -> float:
    """Mean of absolute values — simple amplitude estimate."""
    return float(np.mean(np.abs(signal)))


def compute_variance(signal: np.ndarray) -> float:
    """Signal variance — spread around mean."""
    return float(np.var(signal))


# ── Frequency-domain features ─────────────────────────────────────────────

def compute_spectral_entropy(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE
) -> float:
    """
    Spectral Entropy — measures how spread out the frequency content is.
    Pure sinusoid (one frequency) → low entropy.
    Broadband noise or many harmonics → high entropy.

    Uses normalised PSD as probability distribution for entropy calculation.

    Returns:
        spectral entropy (nats)
    """
    freqs, magnitude, _ = compute_fft(signal, sampling_rate)
    power = magnitude ** 2
    power_sum = power.sum()

    if power_sum < 1e-12:
        return 0.0

    prob = power / power_sum
    # Shannon entropy — ignore zero-probability bins
    prob = prob[prob > 1e-12]
    return float(-np.sum(prob * np.log(prob)))


def compute_spectral_centroid(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE
) -> float:
    """
    Spectral Centroid — frequency-weighted mean of the spectrum.
    Pure 50 Hz → centroid ≈ 50 Hz.
    Harmonic distortion shifts centroid higher.

    Returns:
        centroid frequency in Hz
    """
    freqs, magnitude, _ = compute_fft(signal, sampling_rate)
    mag_sum = magnitude.sum()

    if mag_sum < 1e-12:
        return 0.0

    return float(np.sum(freqs * magnitude) / mag_sum)


def compute_spectral_bandwidth(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE
) -> float:
    """
    Spectral Bandwidth — weighted standard deviation around centroid.
    Narrow for clean signals, wide for distorted ones.

    Returns:
        bandwidth in Hz
    """
    freqs, magnitude, _ = compute_fft(signal, sampling_rate)
    centroid = compute_spectral_centroid(signal, sampling_rate)
    mag_sum  = magnitude.sum()

    if mag_sum < 1e-12:
        return 0.0

    return float(np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / mag_sum))


def compute_spectral_flatness(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE
) -> float:
    """
    Spectral Flatness (Wiener entropy) — ratio of geometric to arithmetic mean of PSD.
    Pure tone → 0 (very tonal, not flat).
    White noise → 1 (perfectly flat).
    Detects how tone-like vs. noise-like the signal is.

    Returns:
        flatness value in [0, 1]
    """
    freqs, magnitude, _ = compute_fft(signal, sampling_rate)
    power = magnitude ** 2 + 1e-12

    geom_mean  = np.exp(np.mean(np.log(power)))
    arith_mean = np.mean(power)

    if arith_mean < 1e-12:
        return 0.0

    return float(np.clip(geom_mean / arith_mean, 0.0, 1.0))


def compute_harmonic_energy_ratio(
    signal: np.ndarray,
    fundamental: float = SIGNAL_FREQ,
    sampling_rate: int = SAMPLING_RATE,
    orders: list = None
) -> float:
    """
    Ratio of harmonic energy (3rd, 5th, ...) to total spectral energy.
    Normal signal → close to 0.
    Heavy harmonic distortion → approaches 1.

    Returns:
        harmonic energy ratio in [0, 1]
    """
    if orders is None:
        from config import HARMONIC_ORDERS
        orders = HARMONIC_ORDERS

    freqs, magnitude, _ = compute_fft(signal, sampling_rate)
    total_energy = np.sum(magnitude ** 2)

    if total_energy < 1e-12:
        return 0.0

    harmonic_amp = get_harmonic_amplitudes(signal, fundamental, orders, sampling_rate)
    harmonic_energy = sum(harmonic_amp[h] ** 2 for h in orders)

    return float(np.clip(harmonic_energy / total_energy, 0.0, 1.0))


def compute_fundamental_amplitude(
    signal: np.ndarray,
    fundamental: float = SIGNAL_FREQ,
    sampling_rate: int = SAMPLING_RATE
) -> float:
    """
    Amplitude of the fundamental frequency component.
    Drops during voltage sag, rises during swell.
    """
    amps = get_harmonic_amplitudes(signal, fundamental, [1], sampling_rate)
    return float(amps.get(1, 0.0))


def compute_peak_count(
    signal: np.ndarray,
    height_factor: float = 0.5
) -> int:
    """
    Count number of peaks above height_factor * max_amplitude.
    Normal 50 Hz → 50 peaks per second.
    Transient spikes add extra high-amplitude peaks.

    Returns:
        number of peaks found
    """
    threshold = height_factor * np.max(np.abs(signal))
    peaks, _  = find_peaks(signal, height=threshold)
    return int(len(peaks))


# ── Voltage quality features ──────────────────────────────────────────────

def compute_voltage_unbalance(signal: np.ndarray) -> float:
    """
    Proxy for voltage unbalance using half-cycle RMS variation.
    Split signal into positive/negative half-cycles and compare.

    Returns:
        unbalance ratio — 0 for perfect balance
    """
    pos_half = signal[signal > 0]
    neg_half = signal[signal < 0]

    if len(pos_half) == 0 or len(neg_half) == 0:
        return 0.0

    rms_pos = np.sqrt(np.mean(pos_half ** 2))
    rms_neg = np.sqrt(np.mean(neg_half ** 2))

    avg = (rms_pos + rms_neg) / 2.0
    if avg < 1e-12:
        return 0.0

    return float(abs(rms_pos - rms_neg) / avg)


def compute_waveform_deviation(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE
) -> float:
    """
    RMS deviation of the signal from a pure 50 Hz sinusoid of the same amplitude.
    Zero for a perfect sinusoid; increases with any distortion.

    Returns:
        normalised deviation (0 to ∞, lower = cleaner)
    """
    from config import NUM_SAMPLES, SIGNAL_DURATION
    t = np.linspace(0, SIGNAL_DURATION, len(signal), endpoint=False)

    # Reconstruct a reference pure sine at the same RMS
    rms_sig = compute_rms(signal)
    ref     = (rms_sig * np.sqrt(2)) * np.sin(2 * np.pi * SIGNAL_FREQ * t)

    deviation = np.sqrt(np.mean((signal - ref) ** 2))
    return float(deviation / (rms_sig + 1e-12))
"""
Layer 2 — DSP Analysis
fft_analyzer.py

FFT-based frequency domain analysis for power grid signals.
Computes magnitude spectrum, phase, dominant frequency,
and per-harmonic amplitudes.
"""

import numpy as np
from config import SAMPLING_RATE, SIGNAL_FREQ, NUM_SAMPLES, HARMONIC_ORDERS


def compute_fft(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE
) -> tuple:
    """
    Compute single-sided FFT magnitude spectrum.

    Args:
        signal:        input waveform (NUM_SAMPLES,)
        sampling_rate: samples per second

    Returns:
        freqs:     frequency axis in Hz  (N/2+1,)
        magnitude: amplitude spectrum     (N/2+1,)
        phase:     phase spectrum in rad  (N/2+1,)
    """
    n = len(signal)
    fft_vals  = np.fft.rfft(signal)
    freqs     = np.fft.rfftfreq(n, d=1.0 / sampling_rate)

    # two-sided → single-sided scaling
    magnitude = (2.0 / n) * np.abs(fft_vals)
    magnitude[0]  /= 2   # DC bin — no doubling
    magnitude[-1] /= 2   # Nyquist bin — no doubling

    phase = np.angle(fft_vals)

    return freqs, magnitude, phase


def get_dominant_frequency(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    freq_min: float = 1.0
) -> float:
    """
    Return the frequency (Hz) with the highest spectral magnitude,
    ignoring DC (below freq_min).

    Args:
        signal:       input waveform
        sampling_rate: samples per second
        freq_min:     minimum frequency to consider (ignore DC)

    Returns:
        dominant frequency in Hz
    """
    freqs, magnitude, _ = compute_fft(signal, sampling_rate)
    mask = freqs >= freq_min
    dominant_idx = np.argmax(magnitude[mask])
    return float(freqs[mask][dominant_idx])


def get_harmonic_amplitudes(
    signal: np.ndarray,
    fundamental: float = SIGNAL_FREQ,
    orders: list = None,
    sampling_rate: int = SAMPLING_RATE,
    bin_tolerance: int = 2
) -> dict:
    """
    Extract the amplitude at each harmonic order of the fundamental.

    Args:
        signal:        input waveform
        fundamental:   fundamental frequency in Hz (default 50 Hz)
        orders:        list of harmonic orders to extract
        sampling_rate: samples per second
        bin_tolerance: ± bins around the target frequency to search

    Returns:
        dict mapping order → amplitude  e.g. {1: 1.0, 3: 0.06, 5: 0.04, ...}
    """
    if orders is None:
        orders = [1] + HARMONIC_ORDERS

    freqs, magnitude, _ = compute_fft(signal, sampling_rate)
    freq_resolution = freqs[1] - freqs[0]  # Hz per bin

    result = {}
    for order in orders:
        target_freq = fundamental * order
        target_bin  = int(round(target_freq / freq_resolution))

        lo = max(0, target_bin - bin_tolerance)
        hi = min(len(magnitude) - 1, target_bin + bin_tolerance)

        result[order] = float(np.max(magnitude[lo:hi + 1]))

    return result


def compute_power_spectral_density(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE
) -> tuple:
    """
    Estimate the Power Spectral Density (PSD) using Welch's method.

    Args:
        signal:        input waveform
        sampling_rate: samples per second

    Returns:
        freqs: frequency axis (Hz)
        psd:   power spectral density (V²/Hz)
    """
    from scipy.signal import welch
    nperseg = min(1024, len(signal))
    freqs, psd = welch(signal, fs=sampling_rate, nperseg=nperseg)
    return freqs, psd


def detect_frequency_deviation(
    signal: np.ndarray,
    nominal: float = SIGNAL_FREQ,
    sampling_rate: int = SAMPLING_RATE,
    tolerance: float = 0.5
) -> dict:
    """
    Detect if grid frequency has deviated from nominal (50 Hz).

    Args:
        signal:       input waveform
        nominal:      expected fundamental frequency in Hz
        sampling_rate: samples per second
        tolerance:    acceptable deviation in Hz (default ±0.5 Hz)

    Returns:
        dict with keys: detected_freq, deviation_hz, is_deviated
    """
    detected = get_dominant_frequency(signal, sampling_rate)
    deviation = detected - nominal

    return {
        "detected_freq":  round(detected, 3),
        "deviation_hz":   round(deviation, 3),
        "is_deviated":    abs(deviation) > tolerance,
    }


def fft_summary(signal: np.ndarray, sampling_rate: int = SAMPLING_RATE) -> dict:
    """
    Full FFT summary for a signal — dominant freq, harmonics, PSD peak.
    Convenient single-call wrapper for dashboard and GenAI layer.

    Returns:
        dict with keys: dominant_freq, harmonic_amplitudes,
                        freq_deviation, fundamental_amplitude
    """
    harmonic_amps = get_harmonic_amplitudes(signal, sampling_rate=sampling_rate)
    freq_dev      = detect_frequency_deviation(signal, sampling_rate=sampling_rate)

    return {
        "dominant_freq":         freq_dev["detected_freq"],
        "deviation_hz":          freq_dev["deviation_hz"],
        "is_deviated":           freq_dev["is_deviated"],
        "fundamental_amplitude": harmonic_amps.get(1, 0.0),
        "harmonic_amplitudes":   harmonic_amps,
    }
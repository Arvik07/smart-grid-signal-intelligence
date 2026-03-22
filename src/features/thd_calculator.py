"""
Layer 3 — Feature Extraction
thd_calculator.py

Total Harmonic Distortion (THD) and related power quality metrics.
THD is the primary indicator of harmonic distortion in power systems.
IEEE standard definition: THD = sqrt(sum(Vn^2)) / V1
"""

import numpy as np
from config import SIGNAL_FREQ, SAMPLING_RATE, HARMONIC_ORDERS
from src.dsp.fft_analyzer import get_harmonic_amplitudes


def compute_thd(
    signal: np.ndarray,
    fundamental: float = SIGNAL_FREQ,
    sampling_rate: int = SAMPLING_RATE,
    orders: list = None
) -> float:
    """
    Compute Total Harmonic Distortion (THD) as a percentage.

    THD(%) = 100 * sqrt(V3² + V5² + V7² + ...) / V1

    Args:
        signal:       input waveform
        fundamental:  fundamental frequency in Hz
        sampling_rate: samples per second
        orders:       harmonic orders to include (default: HARMONIC_ORDERS)

    Returns:
        THD as a percentage (e.g. 5.2 means 5.2%)
    """
    if orders is None:
        orders = HARMONIC_ORDERS

    all_orders   = [1] + list(orders)
    harmonic_amp = get_harmonic_amplitudes(signal, fundamental, all_orders, sampling_rate)

    V1 = harmonic_amp.get(1, 1e-12)
    if V1 < 1e-12:
        return 0.0

    harmonic_power = sum(harmonic_amp[h] ** 2 for h in orders if h in harmonic_amp)
    thd = 100.0 * np.sqrt(harmonic_power) / V1
    return round(thd, 4)


def compute_thd_f(
    signal: np.ndarray,
    fundamental: float = SIGNAL_FREQ,
    sampling_rate: int = SAMPLING_RATE,
    orders: list = None
) -> float:
    """
    Compute THD-F (THD relative to fundamental) — same as compute_thd().
    Alias kept for clarity when reading IEEE power quality standards.
    """
    return compute_thd(signal, fundamental, sampling_rate, orders)


def compute_thd_r(
    signal: np.ndarray,
    fundamental: float = SIGNAL_FREQ,
    sampling_rate: int = SAMPLING_RATE,
    orders: list = None
) -> float:
    """
    Compute THD-R (THD relative to RMS of total signal).

    THD-R(%) = 100 * sqrt(V3² + V5² + ...) / Vrms_total

    Args:
        signal:       input waveform
        fundamental:  fundamental frequency in Hz
        sampling_rate: samples per second

    Returns:
        THD-R as a percentage
    """
    if orders is None:
        orders = HARMONIC_ORDERS

    all_orders   = [1] + list(orders)
    harmonic_amp = get_harmonic_amplitudes(signal, fundamental, all_orders, sampling_rate)

    vrms_total = np.sqrt(np.mean(signal ** 2))
    if vrms_total < 1e-12:
        return 0.0

    harmonic_power = sum(harmonic_amp[h] ** 2 for h in orders if h in harmonic_amp)
    thd_r = 100.0 * np.sqrt(harmonic_power) / vrms_total
    return round(thd_r, 4)


def compute_individual_harmonic_distortion(
    signal: np.ndarray,
    fundamental: float = SIGNAL_FREQ,
    sampling_rate: int = SAMPLING_RATE,
    orders: list = None
) -> dict:
    """
    Compute Individual Harmonic Distortion (IHD) for each harmonic order.

    IHD_n(%) = 100 * Vn / V1

    Args:
        signal:       input waveform
        fundamental:  fundamental frequency in Hz
        sampling_rate: samples per second
        orders:       harmonic orders to evaluate

    Returns:
        dict mapping harmonic order → IHD percentage
        e.g. {3: 6.2, 5: 4.1, 7: 2.8, ...}
    """
    if orders is None:
        orders = HARMONIC_ORDERS

    all_orders   = [1] + list(orders)
    harmonic_amp = get_harmonic_amplitudes(signal, fundamental, all_orders, sampling_rate)

    V1 = harmonic_amp.get(1, 1e-12)
    if V1 < 1e-12:
        return {h: 0.0 for h in orders}

    return {
        h: round(100.0 * harmonic_amp.get(h, 0.0) / V1, 4)
        for h in orders
    }


def compute_power_factor_distortion(thd_percent: float) -> float:
    """
    Estimate the distortion power factor from THD.

    DPF = 1 / sqrt(1 + (THD/100)^2)

    Args:
        thd_percent: THD as a percentage

    Returns:
        Distortion power factor (0 to 1)
    """
    thd_pu = thd_percent / 100.0
    return round(1.0 / np.sqrt(1.0 + thd_pu ** 2), 6)


def thd_summary(
    signal: np.ndarray,
    fundamental: float = SIGNAL_FREQ,
    sampling_rate: int = SAMPLING_RATE
) -> dict:
    """
    Full THD diagnostic summary for a signal.
    Single-call wrapper used by feature pipeline and GenAI layer.

    Returns:
        dict with keys: thd_percent, thd_r_percent, ihd_per_order,
                        distortion_power_factor, dominant_harmonic
    """
    thd     = compute_thd(signal, fundamental, sampling_rate)
    thd_r   = compute_thd_r(signal, fundamental, sampling_rate)
    ihd     = compute_individual_harmonic_distortion(signal, fundamental, sampling_rate)
    dpf     = compute_power_factor_distortion(thd)

    dominant_harmonic = max(ihd, key=ihd.get) if ihd else None

    return {
        "thd_percent":             thd,
        "thd_r_percent":           thd_r,
        "ihd_per_order":           ihd,
        "distortion_power_factor": dpf,
        "dominant_harmonic":       dominant_harmonic,
    }
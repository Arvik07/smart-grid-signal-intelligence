"""
Layer 1 — Signal Simulation
signal_generator.py

Generates synthetic 50 Hz power grid signals for all fault classes.
Each signal is a 1-second window sampled at 10 kHz (10,000 samples).
"""

import numpy as np
from config import (
    SAMPLING_RATE, SIGNAL_FREQ, SIGNAL_DURATION,
    NUM_SAMPLES, AMPLITUDE, FAULT_TYPES, HARMONIC_ORDERS
)


def _time_axis() -> np.ndarray:
    """Return the time vector for one signal window."""
    return np.linspace(0, SIGNAL_DURATION, NUM_SAMPLES, endpoint=False)


# ── Individual signal generators ──────────────────────────────────────────

def generate_normal(amplitude: float = AMPLITUDE) -> np.ndarray:
    """
    Pure 50 Hz sinusoid — the healthy baseline.
        x(t) = A * sin(2π * 50 * t)
    """
    t = _time_axis()
    return amplitude * np.sin(2 * np.pi * SIGNAL_FREQ * t)


def generate_harmonic_distortion(
    fundamental_amp: float = AMPLITUDE,
    harmonic_amps: dict = None
) -> np.ndarray:
    """
    Fundamental + odd harmonics (3rd, 5th, 7th, 9th, 11th).
    Default harmonic amplitude = 0.2 / harmonic_order  (decays naturally).
    """
    t = _time_axis()
    signal = fundamental_amp * np.sin(2 * np.pi * SIGNAL_FREQ * t)

    if harmonic_amps is None:
        harmonic_amps = {h: 0.2 / h for h in HARMONIC_ORDERS}

    for order, amp in harmonic_amps.items():
        signal += amp * np.sin(2 * np.pi * SIGNAL_FREQ * order * t)

    return signal


def generate_voltage_sag(
    sag_depth: float = 0.4,
    sag_start: float = 0.3,
    sag_end: float = 0.7
) -> np.ndarray:
    """
    Voltage sag: amplitude drops to (1 - sag_depth) during [sag_start, sag_end].
    sag_depth=0.4 means voltage falls to 60% of nominal.
    """
    t = _time_axis()
    signal = AMPLITUDE * np.sin(2 * np.pi * SIGNAL_FREQ * t)
    mask = (t >= sag_start) & (t <= sag_end)
    signal[mask] *= (1.0 - sag_depth)
    return signal


def generate_voltage_swell(
    swell_magnitude: float = 0.3,
    swell_start: float = 0.3,
    swell_end: float = 0.7
) -> np.ndarray:
    """
    Voltage swell: amplitude rises to (1 + swell_magnitude) during window.
    swell_magnitude=0.3 means voltage rises to 130% of nominal.
    """
    t = _time_axis()
    signal = AMPLITUDE * np.sin(2 * np.pi * SIGNAL_FREQ * t)
    mask = (t >= swell_start) & (t <= swell_end)
    signal[mask] *= (1.0 + swell_magnitude)
    return signal


def generate_transient_spike(
    n_spikes: int = 3,
    spike_amplitude: float = 2.5,
    spike_width: int = 10
) -> np.ndarray:
    """
    Short-duration high-amplitude transient spikes overlaid on the fundamental.
    """
    t = _time_axis()
    signal = AMPLITUDE * np.sin(2 * np.pi * SIGNAL_FREQ * t)

    rng = np.random.default_rng(seed=42)
    spike_positions = rng.integers(spike_width, NUM_SAMPLES - spike_width, size=n_spikes)

    for pos in spike_positions:
        spike = np.zeros(NUM_SAMPLES)
        spike[pos - spike_width // 2: pos + spike_width // 2] = spike_amplitude
        signal += spike

    return signal


def generate_frequency_deviation(freq_offset: float = 2.5) -> np.ndarray:
    """
    Signal at deviated frequency (50 ± freq_offset Hz).
    Models grid frequency instability.
    """
    t = _time_axis()
    deviated_freq = SIGNAL_FREQ + freq_offset
    return AMPLITUDE * np.sin(2 * np.pi * deviated_freq * t)


# ── Dispatch table ─────────────────────────────────────────────────────────

_GENERATORS = {
    0: generate_normal,
    1: generate_harmonic_distortion,
    2: generate_voltage_sag,
    3: generate_voltage_swell,
    4: generate_transient_spike,
    5: generate_frequency_deviation,
}


def generate_signal(fault_type: int = 0, **kwargs) -> np.ndarray:
    """
    Generate a signal for the given fault_type label.

    Args:
        fault_type: integer key from FAULT_TYPES (0-5)
        **kwargs:   passed to the underlying generator for custom params

    Returns:
        np.ndarray of shape (NUM_SAMPLES,)
    """
    if fault_type not in _GENERATORS:
        raise ValueError(
            f"Unknown fault_type {fault_type}. "
            f"Valid types: {list(FAULT_TYPES.keys())}"
        )
    return _GENERATORS[fault_type](**kwargs)


def generate_dataset(
    n_per_class: int = 200,
    add_noise: bool = True,
    noise_std: float = 0.05,
    augment: bool = True,
    seed: int = 42
) -> tuple:
    """
    Generate a balanced dataset with n_per_class samples per fault type.
    Each sample gets randomised fault parameters + randomised noise level
    so the classifier learns robust patterns rather than memorising templates.

    Args:
        n_per_class: samples per fault class
        add_noise:   add Gaussian noise to every sample
        noise_std:   base noise std — actual std is randomised ±50% per sample
        augment:     randomise fault parameters per sample for variety
        seed:        reproducibility seed

    Returns:
        X: np.ndarray shape (n_total, NUM_SAMPLES)
        y: np.ndarray shape (n_total,) — integer labels
    """
    rng = np.random.default_rng(seed=seed)
    X_list, y_list = [], []

    for fault_type in FAULT_TYPES:
        for _ in range(n_per_class):

            # Randomise fault parameters so each sample is slightly different
            if augment:
                if fault_type == 2:   # voltage_sag
                    depth = rng.uniform(0.2, 0.6)
                    start = rng.uniform(0.1, 0.4)
                    end   = rng.uniform(0.5, 0.9)
                    signal = generate_signal(fault_type, sag_depth=depth,
                                             sag_start=start, sag_end=end)
                elif fault_type == 3:  # voltage_swell
                    mag   = rng.uniform(0.1, 0.5)
                    start = rng.uniform(0.1, 0.4)
                    end   = rng.uniform(0.5, 0.9)
                    signal = generate_signal(fault_type, swell_magnitude=mag,
                                             swell_start=start, swell_end=end)
                elif fault_type == 4:  # transient_spike
                    n_spk = int(rng.integers(1, 6))
                    amp   = rng.uniform(1.5, 3.5)
                    signal = generate_signal(fault_type,
                                             n_spikes=n_spk, spike_amplitude=amp)
                elif fault_type == 5:  # frequency_deviation
                    offset = rng.choice([-1, 1]) * rng.uniform(0.5, 3.5)
                    signal = generate_signal(fault_type, freq_offset=offset)
                else:
                    signal = generate_signal(fault_type)
            else:
                signal = generate_signal(fault_type)

            # Randomise noise level per sample (SNR between 20–40 dB equivalent)
            if add_noise:
                sample_noise_std = noise_std * rng.uniform(0.5, 2.0)
                signal = signal + rng.normal(0, sample_noise_std, size=NUM_SAMPLES)

            X_list.append(signal)
            y_list.append(fault_type)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    idx = rng.permutation(len(y))
    return X[idx], y[idx]
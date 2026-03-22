"""
Layer 1 — Signal Simulation
fault_injector.py

Injects faults into an existing clean signal.
Used for augmentation — take a normal signal and corrupt it on the fly.
"""

import numpy as np
from config import SAMPLING_RATE, SIGNAL_FREQ, NUM_SAMPLES, AMPLITUDE, HARMONIC_ORDERS


def inject_harmonic(signal: np.ndarray, orders: list = None, scale: float = 0.15) -> np.ndarray:
    """
    Add harmonic components to an existing signal.

    Args:
        signal: input waveform (NUM_SAMPLES,)
        orders: list of harmonic orders to inject. Defaults to HARMONIC_ORDERS.
        scale:  amplitude scale for each harmonic relative to fundamental

    Returns:
        distorted signal
    """
    if orders is None:
        orders = HARMONIC_ORDERS

    t = np.linspace(0, 1.0, NUM_SAMPLES, endpoint=False)
    corrupted = signal.copy()

    for order in orders:
        corrupted += (scale / order) * np.sin(2 * np.pi * SIGNAL_FREQ * order * t)

    return corrupted


def inject_sag(
    signal: np.ndarray,
    depth: float = None,
    start_frac: float = None,
    end_frac: float = None,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Apply a randomised voltage sag to a signal window.

    Args:
        signal:     input waveform
        depth:      sag depth 0-1. If None, randomly chosen in [0.2, 0.6]
        start_frac: sag start as fraction of window. If None, random in [0.1, 0.4]
        end_frac:   sag end as fraction of window.   If None, random in [0.5, 0.9]
        rng:        numpy Generator for reproducibility

    Returns:
        sagged signal
    """
    if rng is None:
        rng = np.random.default_rng()

    depth      = depth      if depth      is not None else rng.uniform(0.2, 0.6)
    start_frac = start_frac if start_frac is not None else rng.uniform(0.1, 0.4)
    end_frac   = end_frac   if end_frac   is not None else rng.uniform(0.5, 0.9)

    t = np.linspace(0, 1.0, NUM_SAMPLES, endpoint=False)
    corrupted = signal.copy()
    mask = (t >= start_frac) & (t <= end_frac)
    corrupted[mask] *= (1.0 - depth)
    return corrupted


def inject_swell(
    signal: np.ndarray,
    magnitude: float = None,
    start_frac: float = None,
    end_frac: float = None,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Apply a randomised voltage swell.

    Args:
        magnitude: swell magnitude above 1.0. If None, random in [0.1, 0.4]
    """
    if rng is None:
        rng = np.random.default_rng()

    magnitude  = magnitude  if magnitude  is not None else rng.uniform(0.1, 0.4)
    start_frac = start_frac if start_frac is not None else rng.uniform(0.1, 0.4)
    end_frac   = end_frac   if end_frac   is not None else rng.uniform(0.5, 0.9)

    t = np.linspace(0, 1.0, NUM_SAMPLES, endpoint=False)
    corrupted = signal.copy()
    mask = (t >= start_frac) & (t <= end_frac)
    corrupted[mask] *= (1.0 + magnitude)
    return corrupted


def inject_transient(
    signal: np.ndarray,
    n_spikes: int = None,
    spike_amp: float = None,
    spike_width: int = 10,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Inject random transient spikes into a signal.

    Args:
        n_spikes:    number of spikes. If None, random in [1, 5]
        spike_amp:   spike amplitude. If None, random in [1.5, 3.5]
        spike_width: width of each spike in samples
    """
    if rng is None:
        rng = np.random.default_rng()

    n_spikes  = n_spikes  if n_spikes  is not None else int(rng.integers(1, 6))
    spike_amp = spike_amp if spike_amp is not None else rng.uniform(1.5, 3.5)

    corrupted = signal.copy()
    positions = rng.integers(spike_width, NUM_SAMPLES - spike_width, size=n_spikes)

    for pos in positions:
        half = spike_width // 2
        corrupted[pos - half: pos + half] += spike_amp

    return corrupted


def inject_frequency_deviation(
    signal: np.ndarray,
    offset: float = None,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Replace the signal with a version at a slightly deviated frequency.

    Args:
        offset: Hz offset from 50 Hz. If None, random in [-3, 3] (non-zero)
    """
    if rng is None:
        rng = np.random.default_rng()

    if offset is None:
        offset = rng.choice([-1, 1]) * rng.uniform(1.0, 3.0)

    t = np.linspace(0, 1.0, NUM_SAMPLES, endpoint=False)
    return AMPLITUDE * np.sin(2 * np.pi * (SIGNAL_FREQ + offset) * t)


# ── Augmentation pipeline ─────────────────────────────────────────────────

_INJECTORS = {
    1: inject_harmonic,
    2: inject_sag,
    3: inject_swell,
    4: inject_transient,
    5: inject_frequency_deviation,
}


def augment_signal(
    signal: np.ndarray,
    fault_type: int,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Apply the fault injection corresponding to fault_type onto a clean signal.
    fault_type=0 (normal) returns the signal unchanged.
    """
    if fault_type == 0:
        return signal.copy()

    if fault_type not in _INJECTORS:
        raise ValueError(f"Unknown fault_type: {fault_type}")

    if rng is None:
        rng = np.random.default_rng()

    return _INJECTORS[fault_type](signal, rng=rng)
"""
Layer 1 — Signal Simulation
noise_utils.py

Realistic noise models for power grid signal corruption.
Provides Gaussian white noise, pink noise (1/f), and impulse noise.
"""

import numpy as np
from config import NUM_SAMPLES, SAMPLING_RATE


def add_gaussian_noise(
    signal: np.ndarray,
    snr_db: float = 30.0,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Add Additive White Gaussian Noise (AWGN) at a specified SNR.

    Args:
        signal: clean input signal
        snr_db: signal-to-noise ratio in dB. Higher = cleaner.
                Typical power grid monitoring: 20-40 dB
        rng:    random generator for reproducibility

    Returns:
        noisy signal
    """
    if rng is None:
        rng = np.random.default_rng()

    signal_power = np.mean(signal ** 2)
    snr_linear   = 10 ** (snr_db / 10.0)
    noise_power  = signal_power / snr_linear
    noise        = rng.normal(0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise


def add_pink_noise(
    signal: np.ndarray,
    scale: float = 0.02,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Add pink noise (1/f noise) — common in electronic measurement systems.
    Generated via frequency-domain shaping of white noise.

    Args:
        signal: clean input signal
        scale:  amplitude scale of the pink noise
        rng:    random generator

    Returns:
        signal with pink noise added
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(signal)
    white = rng.standard_normal(n)

    # Shape in frequency domain: multiply FFT by 1/sqrt(f)
    fft_white  = np.fft.rfft(white)
    freqs      = np.fft.rfftfreq(n, d=1.0 / SAMPLING_RATE)
    freqs[0]   = 1.0  # avoid divide-by-zero at DC

    pink_filter      = 1.0 / np.sqrt(freqs)
    pink_filter[0]   = 0.0  # remove DC component

    fft_pink   = fft_white * pink_filter
    pink_noise = np.fft.irfft(fft_pink, n=n)

    # Normalise to desired scale
    pink_noise = pink_noise / np.std(pink_noise) * scale

    return signal + pink_noise


def add_impulse_noise(
    signal: np.ndarray,
    probability: float = 0.001,
    amplitude: float = 3.0,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Add impulse (salt-and-pepper) noise — models measurement glitches.

    Args:
        signal:      clean input signal
        probability: fraction of samples corrupted (default: 0.1%)
        amplitude:   magnitude of impulse relative to signal amplitude
        rng:         random generator

    Returns:
        signal with impulse noise
    """
    if rng is None:
        rng = np.random.default_rng()

    corrupted = signal.copy()
    mask      = rng.random(size=signal.shape) < probability
    signs     = rng.choice([-1, 1], size=mask.sum())
    corrupted[mask] += signs * amplitude
    return corrupted


def add_combined_noise(
    signal: np.ndarray,
    snr_db: float = 30.0,
    pink_scale: float = 0.01,
    impulse_prob: float = 0.0005,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Apply Gaussian + pink + impulse noise in sequence.
    Realistic model for a power grid sensor measurement.

    Args:
        signal:       clean input signal
        snr_db:       AWGN SNR in dB
        pink_scale:   pink noise amplitude scale
        impulse_prob: impulse noise probability per sample

    Returns:
        corrupted signal
    """
    if rng is None:
        rng = np.random.default_rng()

    noisy = add_gaussian_noise(signal, snr_db=snr_db, rng=rng)
    noisy = add_pink_noise(noisy, scale=pink_scale, rng=rng)
    noisy = add_impulse_noise(noisy, probability=impulse_prob, rng=rng)
    return noisy


def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """
    Compute the actual SNR in dB between a clean and noisy signal.

    Args:
        clean: reference clean signal
        noisy: degraded signal

    Returns:
        SNR in dB
    """
    noise        = noisy - clean
    signal_power = np.mean(clean ** 2)
    noise_power  = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')

    return 10 * np.log10(signal_power / noise_power)
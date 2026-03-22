"""
Layer 2 — DSP Analysis
spectrogram.py

Short-Time Fourier Transform (STFT) and spectrogram computation.
Provides time-frequency representations for detecting non-stationary
faults like transient spikes and time-varying voltage sags.
"""

import numpy as np
from scipy.signal import stft, spectrogram as scipy_spectrogram
from config import SAMPLING_RATE, SIGNAL_FREQ


def compute_stft(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: int = None,
    nfft: int = None
) -> tuple:
    """
    Compute the Short-Time Fourier Transform (STFT).

    Args:
        signal:       input waveform
        sampling_rate: samples per second
        window:       window function — hann, hamming, blackman
        nperseg:      samples per STFT segment
        noverlap:     overlap between segments (default: nperseg // 2)
        nfft:         FFT size (default: nperseg)

    Returns:
        freqs:  frequency axis in Hz    (F,)
        times:  time axis in seconds    (T,)
        Zxx:    complex STFT matrix     (F, T)
    """
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = nperseg

    freqs, times, Zxx = stft(
        signal,
        fs=sampling_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft
    )
    return freqs, times, Zxx


def compute_spectrogram(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: int = None
) -> tuple:
    """
    Compute power spectrogram (|STFT|²) in dB scale.

    Args:
        signal:       input waveform
        sampling_rate: samples per second
        window:       window function
        nperseg:      samples per segment
        noverlap:     overlap samples

    Returns:
        freqs:   frequency axis in Hz   (F,)
        times:   time axis in seconds   (T,)
        Sxx_db:  spectrogram in dB      (F, T)
    """
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, times, Sxx = scipy_spectrogram(
        signal,
        fs=sampling_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap
    )

    # Convert to dB — add small epsilon to avoid log(0)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    return freqs, times, Sxx_db


def compute_stft_magnitude(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    nperseg: int = 256,
    noverlap: int = None
) -> tuple:
    """
    Compute STFT and return magnitude (not power, not dB).
    Useful for feature extraction and ML input.

    Returns:
        freqs:     frequency axis in Hz  (F,)
        times:     time axis in seconds  (T,)
        magnitude: |STFT| matrix         (F, T)
    """
    freqs, times, Zxx = compute_stft(signal, sampling_rate, nperseg=nperseg, noverlap=noverlap)
    return freqs, times, np.abs(Zxx)


def detect_transients(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    nperseg: int = 64,
    energy_threshold_factor: float = 3.0
) -> dict:
    """
    Detect transient events using STFT energy analysis.
    A transient is detected when a time-frame's total energy
    exceeds mean_energy * energy_threshold_factor.

    Args:
        signal:                 input waveform
        sampling_rate:          samples per second
        nperseg:                short window for fine time resolution
        energy_threshold_factor: multiplier above mean to flag as transient

    Returns:
        dict with keys:
            transient_times: list of timestamps (seconds) where transients occur
            n_transients:    count of detected transients
            energy_profile:  per-frame total energy array
    """
    freqs, times, Zxx = compute_stft(signal, sampling_rate, nperseg=nperseg)

    # Total energy per time frame
    energy_profile = np.sum(np.abs(Zxx) ** 2, axis=0)

    mean_energy = np.mean(energy_profile)
    threshold   = mean_energy * energy_threshold_factor

    transient_mask  = energy_profile > threshold
    transient_times = list(times[transient_mask].round(5))

    return {
        "transient_times": transient_times,
        "n_transients":    int(transient_mask.sum()),
        "energy_profile":  energy_profile,
    }


def detect_voltage_event(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    nperseg: int = 256,
    sag_threshold: float = 0.85,
    swell_threshold: float = 1.10
) -> dict:
    """
    Detect voltage sag / swell events using STFT envelope.
    Tracks the RMS amplitude of the fundamental over time.

    Args:
        signal:           input waveform
        sampling_rate:    samples per second
        nperseg:          samples per segment
        sag_threshold:    amplitude below this fraction = sag   (e.g. 0.85 = 85%)
        swell_threshold:  amplitude above this fraction = swell (e.g. 1.10 = 110%)

    Returns:
        dict with keys:
            event_type:       'sag' | 'swell' | 'normal'
            start_time_s:     event start in seconds (or None)
            end_time_s:       event end in seconds   (or None)
            min_amplitude:    minimum normalised amplitude during window
            max_amplitude:    maximum normalised amplitude during window
    """
    freqs, times, Zxx = compute_stft(signal, sampling_rate, nperseg=nperseg)

    # Find the fundamental bin
    freq_resolution = freqs[1] - freqs[0]
    fund_bin = int(round(SIGNAL_FREQ / freq_resolution))
    fund_bin = min(fund_bin, len(freqs) - 1)

    # Amplitude envelope at fundamental
    amp_envelope = np.abs(Zxx[fund_bin, :])
    amp_norm     = amp_envelope / (np.max(amp_envelope) + 1e-12)

    min_amp = float(amp_norm.min())
    max_amp = float(amp_norm.max())

    event_type  = "normal"
    start_time  = None
    end_time    = None

    if min_amp < sag_threshold:
        event_type = "sag"
        sag_frames = np.where(amp_norm < sag_threshold)[0]
        start_time = float(times[sag_frames[0]])
        end_time   = float(times[sag_frames[-1]])

    elif max_amp > swell_threshold:
        event_type = "swell"
        swell_frames = np.where(amp_norm > swell_threshold)[0]
        start_time   = float(times[swell_frames[0]])
        end_time     = float(times[swell_frames[-1]])

    return {
        "event_type":    event_type,
        "start_time_s":  round(start_time, 4) if start_time is not None else None,
        "end_time_s":    round(end_time, 4)   if end_time   is not None else None,
        "min_amplitude": round(min_amp, 4),
        "max_amplitude": round(max_amp, 4),
    }


def stft_to_feature_matrix(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    nperseg: int = 256,
    max_freq_hz: float = 1000.0
) -> np.ndarray:
    """
    Convert STFT magnitude to a 2D feature matrix for ML.
    Truncates to max_freq_hz to focus on relevant frequency range.

    Args:
        signal:       input waveform
        sampling_rate: samples per second
        nperseg:      samples per segment
        max_freq_hz:  keep frequencies up to this value

    Returns:
        feature_matrix: 2D array (n_freq_bins, n_time_frames)
    """
    freqs, times, magnitude = compute_stft_magnitude(signal, sampling_rate, nperseg=nperseg)

    # Truncate to max_freq_hz
    freq_mask = freqs <= max_freq_hz
    return magnitude[freq_mask, :]
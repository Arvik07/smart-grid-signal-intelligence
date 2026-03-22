"""
Layer 2 — DSP Analysis
filter_design.py

FIR and IIR digital filter design and application for power grid signals.
Covers low-pass, high-pass, band-pass, band-stop (notch), and
harmonic-selective filters.
"""

import numpy as np
from scipy import signal as sp
from config import SAMPLING_RATE, SIGNAL_FREQ


# ── FIR Filters ───────────────────────────────────────────────────────────

def design_fir_lowpass(
    cutoff_hz: float,
    num_taps: int = 101,
    sampling_rate: int = SAMPLING_RATE,
    window: str = "hamming"
) -> np.ndarray:
    """
    Design a FIR low-pass filter using window method.

    Args:
        cutoff_hz:    cutoff frequency in Hz
        num_taps:     filter order + 1 (must be odd for Type I)
        sampling_rate: samples per second
        window:       window function — hamming, hann, blackman, kaiser

    Returns:
        b: FIR filter coefficients (num_taps,)
    """
    nyquist  = sampling_rate / 2.0
    cutoff_n = cutoff_hz / nyquist  # normalised [0, 1]
    b = sp.firwin(num_taps, cutoff_n, window=window)
    return b


def design_fir_bandpass(
    lowcut_hz: float,
    highcut_hz: float,
    num_taps: int = 101,
    sampling_rate: int = SAMPLING_RATE,
    window: str = "hamming"
) -> np.ndarray:
    """
    Design a FIR band-pass filter.

    Args:
        lowcut_hz:  lower cutoff frequency in Hz
        highcut_hz: upper cutoff frequency in Hz
        num_taps:   filter order + 1
        sampling_rate: samples per second

    Returns:
        b: FIR filter coefficients
    """
    nyquist = sampling_rate / 2.0
    low_n   = lowcut_hz  / nyquist
    high_n  = highcut_hz / nyquist
    b = sp.firwin(num_taps, [low_n, high_n], pass_zero=False, window=window)
    return b


def design_fir_bandstop(
    lowcut_hz: float,
    highcut_hz: float,
    num_taps: int = 101,
    sampling_rate: int = SAMPLING_RATE,
    window: str = "hamming"
) -> np.ndarray:
    """
    Design a FIR band-stop (notch) filter.

    Args:
        lowcut_hz:  lower edge of stopband in Hz
        highcut_hz: upper edge of stopband in Hz

    Returns:
        b: FIR filter coefficients
    """
    nyquist = sampling_rate / 2.0
    low_n   = lowcut_hz  / nyquist
    high_n  = highcut_hz / nyquist
    b = sp.firwin(num_taps, [low_n, high_n], pass_zero=True, window=window)
    return b


# ── IIR Filters ───────────────────────────────────────────────────────────

def design_iir_butterworth(
    cutoff_hz,
    order: int = 5,
    filter_type: str = "low",
    sampling_rate: int = SAMPLING_RATE
) -> tuple:
    """
    Design a Butterworth IIR filter (maximally flat magnitude).

    Args:
        cutoff_hz:   cutoff Hz — scalar for low/high, [low, high] for band/bandstop
        order:       filter order (higher = sharper rolloff, more ringing)
        filter_type: 'low' | 'high' | 'band' | 'bandstop'
        sampling_rate: samples per second

    Returns:
        b, a: IIR filter coefficients
    """
    nyquist  = sampling_rate / 2.0
    if isinstance(cutoff_hz, (list, tuple)):
        wn = [f / nyquist for f in cutoff_hz]
    else:
        wn = cutoff_hz / nyquist

    b, a = sp.butter(order, wn, btype=filter_type)
    return b, a


def design_iir_notch(
    notch_freq_hz: float,
    quality_factor: float = 30.0,
    sampling_rate: int = SAMPLING_RATE
) -> tuple:
    """
    Design a narrow IIR notch filter to suppress a specific frequency.
    Ideal for removing individual harmonics (e.g. 150 Hz 3rd harmonic).

    Args:
        notch_freq_hz:  frequency to suppress in Hz
        quality_factor: Q-factor — higher Q = narrower notch
        sampling_rate:  samples per second

    Returns:
        b, a: IIR filter coefficients
    """
    w0 = notch_freq_hz / (sampling_rate / 2.0)
    b, a = sp.iirnotch(w0, quality_factor)
    return b, a


# ── Filter Application ────────────────────────────────────────────────────

def apply_fir_filter(
    signal_in: np.ndarray,
    b: np.ndarray,
    zero_phase: bool = True
) -> np.ndarray:
    """
    Apply FIR filter to a signal.

    Args:
        signal_in:  input waveform
        b:          FIR coefficients
        zero_phase: use filtfilt (zero phase delay) if True, else lfilter

    Returns:
        filtered signal
    """
    if zero_phase:
        return sp.filtfilt(b, [1.0], signal_in)
    return sp.lfilter(b, [1.0], signal_in)


def apply_iir_filter(
    signal_in: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    zero_phase: bool = True
) -> np.ndarray:
    """
    Apply IIR filter to a signal.

    Args:
        signal_in:  input waveform
        b, a:       IIR numerator/denominator coefficients
        zero_phase: use filtfilt (zero phase) if True, else lfilter

    Returns:
        filtered signal
    """
    if zero_phase:
        return sp.filtfilt(b, a, signal_in)
    return sp.lfilter(b, a, signal_in)


# ── Ready-made pipeline filters ───────────────────────────────────────────

def remove_harmonics(
    signal_in: np.ndarray,
    orders: list = None,
    fundamental: float = SIGNAL_FREQ,
    sampling_rate: int = SAMPLING_RATE,
    q_factor: float = 30.0
) -> np.ndarray:
    """
    Chain notch filters to remove multiple harmonics in one call.

    Args:
        signal_in:  input waveform
        orders:     harmonic orders to remove (e.g. [3, 5, 7])
        fundamental: fundamental frequency in Hz
        q_factor:   quality factor for each notch

    Returns:
        signal with harmonics removed
    """
    if orders is None:
        orders = [3, 5, 7, 9, 11]

    cleaned = signal_in.copy()
    for order in orders:
        freq = fundamental * order
        b, a = design_iir_notch(freq, q_factor, sampling_rate)
        cleaned = apply_iir_filter(cleaned, b, a)

    return cleaned


def extract_fundamental(
    signal_in: np.ndarray,
    fundamental: float = SIGNAL_FREQ,
    bandwidth_hz: float = 5.0,
    sampling_rate: int = SAMPLING_RATE
) -> np.ndarray:
    """
    Isolate the fundamental frequency component using a narrow BPF.

    Args:
        signal_in:    input waveform
        fundamental:  center frequency in Hz
        bandwidth_hz: ± bandwidth around the fundamental
        sampling_rate: samples per second

    Returns:
        signal containing only the fundamental component
    """
    low  = fundamental - bandwidth_hz
    high = fundamental + bandwidth_hz
    b, a = design_iir_butterworth(
        [low, high], order=4, filter_type="band", sampling_rate=sampling_rate
    )
    return apply_iir_filter(signal_in, b, a)


def denoise_signal(
    signal_in: np.ndarray,
    snr_threshold_hz: float = 2000.0,
    sampling_rate: int = SAMPLING_RATE
) -> np.ndarray:
    """
    Remove high-frequency noise above snr_threshold_hz using a Butterworth LPF.

    Args:
        signal_in:         noisy input signal
        snr_threshold_hz:  cutoff — frequencies above this are attenuated
        sampling_rate:     samples per second

    Returns:
        denoised signal
    """
    b, a = design_iir_butterworth(
        snr_threshold_hz, order=6, filter_type="low", sampling_rate=sampling_rate
    )
    return apply_iir_filter(signal_in, b, a)


def get_filter_frequency_response(
    b: np.ndarray,
    a: np.ndarray = None,
    n_points: int = 512,
    sampling_rate: int = SAMPLING_RATE
) -> tuple:
    """
    Compute frequency response of a filter for visualisation.

    Args:
        b, a:        filter coefficients (a=[1.0] for FIR)
        n_points:    number of frequency points
        sampling_rate: samples per second

    Returns:
        freqs: frequency axis in Hz
        H_db:  magnitude response in dB
    """
    if a is None:
        a = [1.0]

    w, H = sp.freqz(b, a, worN=n_points, fs=sampling_rate)
    H_db = 20 * np.log10(np.abs(H) + 1e-12)
    return w, H_db
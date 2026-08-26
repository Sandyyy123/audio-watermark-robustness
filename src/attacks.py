"""Real-world channel transformations to stress the detector against."""
from __future__ import annotations
import os
import subprocess
import tempfile
import numpy as np
from scipy.signal import resample_poly, butter, sosfilt
from audio_io import write_wav, read_wav


def additive_noise(x: np.ndarray, sr: int, snr_db: float = 20.0) -> np.ndarray:
    rng = np.random.default_rng(1)
    p = np.mean(x ** 2) + 1e-12
    n = rng.standard_normal(len(x))
    n *= np.sqrt(p / (10 ** (snr_db / 10)) / (np.mean(n ** 2) + 1e-12))
    return x + n


def resample_roundtrip(x: np.ndarray, sr: int, down: int = 2) -> np.ndarray:
    y = resample_poly(x, 1, down)
    y = resample_poly(y, down, 1)
    return y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def requantize(x: np.ndarray, sr: int, bits: int = 8) -> np.ndarray:
    q = 2 ** (bits - 1)
    return np.round(np.clip(x, -1, 1) * q) / q


def amplitude_scale(x: np.ndarray, sr: int, gain: float = 0.5) -> np.ndarray:
    return x * gain


def lowpass(x: np.ndarray, sr: int, cutoff_hz: float = 8000.0) -> np.ndarray:
    sos = butter(6, cutoff_hz / (sr / 2), btype="low", output="sos")
    return sosfilt(sos, x)


def highpass(x: np.ndarray, sr: int, cutoff_hz: float = 150.0) -> np.ndarray:
    sos = butter(4, cutoff_hz / (sr / 2), btype="high", output="sos")
    return sosfilt(sos, x)


def time_stretch(x: np.ndarray, sr: int, rate: float = 1.02) -> np.ndarray:
    """Naive resample-based stretch (breaks frame sync -> the hard case)."""
    n = int(len(x) / rate)
    y = resample_poly(x, n, len(x))
    return y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def mp3_roundtrip(x: np.ndarray, sr: int, bitrate: str = "128k") -> np.ndarray:
    """True MP3 encode/decode via ffmpeg if available; else fall back to lowpass+quant."""
    if not _has_ffmpeg():
        return requantize(lowpass(x, sr, 15000.0), sr, 10)
    with tempfile.TemporaryDirectory() as d:
        win = os.path.join(d, "in.wav")
        mp3 = os.path.join(d, "c.mp3")
        wout = os.path.join(d, "out.wav")
        write_wav(win, x, sr)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-i", win,
                        "-b:a", bitrate, mp3], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "quiet", "-i", mp3,
                        "-ar", str(sr), wout], check=True)
        y, _ = read_wav(wout)
        if y.ndim == 2:
            y = y.mean(axis=1)
    return y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def _has_ffmpeg() -> bool:
    from shutil import which
    return which("ffmpeg") is not None


def attack_suite() -> dict:
    return {
        "none": lambda x, sr: x,
        "mp3_128k": mp3_roundtrip,
        "mp3_64k": lambda x, sr: mp3_roundtrip(x, sr, "64k"),
        "resample_22k": resample_roundtrip,
        "noise_30dB": lambda x, sr: additive_noise(x, sr, 30.0),
        "noise_25dB": lambda x, sr: additive_noise(x, sr, 25.0),
        "noise_20dB": lambda x, sr: additive_noise(x, sr, 20.0),
        "requantize_8bit": requantize,
        "gain_0.5x": amplitude_scale,
        "lowpass_8kHz": lowpass,
        "highpass_150Hz": highpass,
        "time_stretch_2pct": time_stretch,
    }

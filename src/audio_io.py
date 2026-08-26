"""WAV I/O and a self-contained synthetic test-signal generator (no external audio needed)."""
from __future__ import annotations
import numpy as np
from scipy.io import wavfile


def read_wav(path: str) -> tuple[np.ndarray, int]:
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        x = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        x = (data.astype(np.float64) - 128.0) / 128.0
    else:
        x = data.astype(np.float64)
    return x, sr


def write_wav(path: str, x: np.ndarray, sr: int) -> None:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -1.0, 1.0)
    wavfile.write(path, sr, (x * 32767.0).astype(np.int16))


def synth_signal(sr: int = 44100, seconds: float = 6.0, seed: int = 7) -> np.ndarray:
    """A music-like broadband signal: harmonic chords + vibrato + noise + envelope."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    x = np.zeros_like(t)
    roots = [220.0, 277.18, 329.63, 246.94]      # rotating chord roots (A, C#, E, B)
    seg = len(t) // len(roots)
    for i, f0 in enumerate(roots):
        s, e = i * seg, (i + 1) * seg if i < len(roots) - 1 else len(t)
        tt = t[s:e]
        vib = 1.0 + 0.003 * np.sin(2 * np.pi * 5.0 * tt)
        for h, amp in enumerate([1.0, 0.5, 0.33, 0.25, 0.18, 0.12], start=1):
            x[s:e] += amp * np.sin(2 * np.pi * f0 * h * vib * tt)
    x += 0.02 * rng.standard_normal(len(t))       # broadband noise floor
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 0.8 * t) ** 2
    x *= env
    x /= (np.max(np.abs(x)) + 1e-9)
    return x * 0.9

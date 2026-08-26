"""
Blind spread-spectrum audio watermarking (magnitude-STFT domain).

Design (classic multiplicative spread spectrum, Cox et al. 1997, adapted to a
mid-band STFT-magnitude carrier with a key-seeded chip sequence):

  * A secret integer key seeds a +/-1 chip sequence c[t,k] over the selected
    time-frequency coefficients.
  * Each payload bit is spread across many coefficients (redundancy -> robustness).
  * Embedding is MULTIPLICATIVE:  |X'| = |X| * (1 + alpha * c * b)
    so the watermark energy scales with the local host magnitude. That is a
    cheap perceptual mask: loud bins carry more watermark, quiet bins carry
    less, which keeps it below the masking threshold and near-inaudible.
  * Detection is BLIND (no original needed). The host spectral envelope is
    estimated by a short moving-average across frequency and subtracted, which
    whitens the host away because the chip sequence is white in the frequency
    index. The residual correlates with the key chips -> per-bit sign recovers
    the payload, and the aggregate matched energy gives a payload-blind
    present/absent detection statistic for ROC / false-positive control.

Only numpy + scipy. WAV-first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.signal import stft, istft
from scipy.ndimage import uniform_filter1d


@dataclass
class WatermarkConfig:
    sr: int = 44100
    n_fft: int = 1024
    hop: int = 512
    band_lo_hz: float = 2000.0     # mid band: good masking, survives MP3 / resample
    band_hi_hz: float = 7000.0
    alpha: float = 0.35            # embedding strength (multiplicative); ~41 dB SNR
    payload_bits: int = 16
    host_est_bins: int = 9         # freq-axis moving average width for host removal
    window: str = "hann"

    @property
    def nperseg(self) -> int:
        return self.n_fft

    @property
    def noverlap(self) -> int:
        return self.n_fft - self.hop


def _band_bin_range(cfg: WatermarkConfig) -> tuple[int, int]:
    freqs = np.fft.rfftfreq(cfg.n_fft, 1.0 / cfg.sr)
    lo = int(np.searchsorted(freqs, cfg.band_lo_hz))
    hi = int(np.searchsorted(freqs, cfg.band_hi_hz))
    return lo, hi


def _chips_and_bitmap(n_frames: int, cfg: WatermarkConfig, key: int):
    """Deterministic +/-1 chip field and payload-bit assignment for the band."""
    lo, hi = _band_bin_range(cfg)
    n_bins = hi - lo
    rng = np.random.default_rng(key)
    chips = rng.choice(np.array([-1.0, 1.0]), size=(n_frames, n_bins))
    # round-robin bit assignment, spread across time and frequency
    n = n_frames * n_bins
    bit_of = (np.arange(n) % cfg.payload_bits).reshape(n_frames, n_bins)
    return lo, hi, chips, bit_of


def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float64)


def embed(x: np.ndarray, payload_bits: np.ndarray, cfg: WatermarkConfig, key: int) -> np.ndarray:
    """Embed a bit array (values in {0,1}) into mono signal x. Returns watermarked signal."""
    x = _to_mono(x)
    peak = np.max(np.abs(x)) + 1e-12
    xn = x / peak

    f, t, Z = stft(xn, fs=cfg.sr, window=cfg.window, nperseg=cfg.nperseg,
                   noverlap=cfg.noverlap, boundary="zeros", padded=True)
    mag = np.abs(Z)            # (n_bins_full, n_frames)
    phase = np.angle(Z)
    n_frames = mag.shape[1]

    lo, hi, chips, bit_of = _chips_and_bitmap(n_frames, cfg, key)
    b = np.where(np.asarray(payload_bits).astype(int) > 0, 1.0, -1.0)  # {0,1} -> {-1,+1}
    b_field = b[bit_of.T]      # (n_bins, n_frames) after transpose alignment
    chip_field = chips.T       # (n_bins, n_frames)

    band = mag[lo:hi, :]
    band_wm = band * (1.0 + cfg.alpha * chip_field * b_field)
    band_wm = np.maximum(band_wm, 0.0)
    mag[lo:hi, :] = band_wm

    Zwm = mag * np.exp(1j * phase)
    _, y = istft(Zwm, fs=cfg.sr, window=cfg.window, nperseg=cfg.nperseg, noverlap=cfg.noverlap)
    y = y[: len(xn)] if len(y) >= len(xn) else np.pad(y, (0, len(xn) - len(y)))
    y = y * peak
    m = np.max(np.abs(y))
    if m > 1.0:
        y = y / m * 0.999
    return y.astype(np.float32)


def _residual(mag: np.ndarray, lo: int, hi: int, cfg: WatermarkConfig) -> np.ndarray:
    """Whiten host envelope: subtract freq-axis moving average, normalize by it."""
    band = mag[lo:hi, :]
    host = uniform_filter1d(band, size=cfg.host_est_bins, axis=0, mode="nearest")
    return (band - host) / (host + 1e-9)


def detect(y: np.ndarray, cfg: WatermarkConfig, key: int) -> dict:
    """
    Blind detect. Returns dict:
      bits      -> recovered payload (0/1) array
      statistic -> payload-blind present/absent detection statistic (z-like)
      per_bit_z -> per-bit correlation z-scores
    """
    y = _to_mono(y)
    peak = np.max(np.abs(y)) + 1e-12
    yn = y / peak
    f, t, Z = stft(yn, fs=cfg.sr, window=cfg.window, nperseg=cfg.nperseg,
                   noverlap=cfg.noverlap, boundary="zeros", padded=True)
    mag = np.abs(Z)
    n_frames = mag.shape[1]

    lo, hi, chips, bit_of = _chips_and_bitmap(n_frames, cfg, key)
    r = _residual(mag, lo, hi, cfg)             # (n_bins, n_frames)
    chip_field = chips.T
    d = r * chip_field                          # correlate against key chips

    bit_flat = bit_of.T.reshape(-1)
    d_flat = d.reshape(-1)
    per_bit_sum = np.zeros(cfg.payload_bits)
    per_bit_cnt = np.zeros(cfg.payload_bits)
    per_bit_sq = np.zeros(cfg.payload_bits)
    np.add.at(per_bit_sum, bit_flat, d_flat)
    np.add.at(per_bit_cnt, bit_flat, 1.0)
    np.add.at(per_bit_sq, bit_flat, d_flat ** 2)

    mean = per_bit_sum / np.maximum(per_bit_cnt, 1)
    var = per_bit_sq / np.maximum(per_bit_cnt, 1) - mean ** 2
    std_err = np.sqrt(np.maximum(var, 1e-12) / np.maximum(per_bit_cnt, 1))
    per_bit_z = mean / (std_err + 1e-12)
    bits = (per_bit_z > 0).astype(int)

    # payload-blind detection statistic: matched energy across all coefficients
    total = d_flat.sum()
    n = d_flat.size
    noise_std = d_flat.std() + 1e-12
    # aggregate matched-filter z using recovered signs (payload-blind magnitude)
    statistic = float(np.sum(np.abs(per_bit_z)) / np.sqrt(cfg.payload_bits))
    return {
        "bits": bits,
        "statistic": statistic,
        "per_bit_z": per_bit_z,
        "raw_corr": float(total / (noise_std * np.sqrt(n))),
    }


def detect_resync(y: np.ndarray, cfg: WatermarkConfig, key: int,
                  rates=None) -> dict:
    """
    Detection with a small resampling-rate search to recover from time
    scaling / desync (tape speed, ASR pitch shift, clock drift). Tries a set of
    candidate stretch factors, resamples, and keeps the hypothesis with the
    strongest detection statistic. This is the standard answer to the
    frame-synchronization problem that naive detectors fail on.
    """
    from scipy.signal import resample_poly
    if rates is None:
        rates = [0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03]
    best = None
    for r in rates:
        up, down = _ratio(r)
        yr = resample_poly(y, up, down) if r != 1.0 else y
        res = detect(yr, cfg, key)
        res["rate"] = r
        if best is None or res["statistic"] > best["statistic"]:
            best = res
    return best


def _ratio(r: float, denom: int = 1000) -> tuple[int, int]:
    return int(round(r * denom)), denom


def ber(true_bits: np.ndarray, rec_bits: np.ndarray) -> float:
    true_bits = np.asarray(true_bits).astype(int)
    rec_bits = np.asarray(rec_bits).astype(int)
    n = min(len(true_bits), len(rec_bits))
    return float(np.mean(true_bits[:n] != rec_bits[:n]))


def snr_db(original: np.ndarray, watermarked: np.ndarray) -> float:
    original = _to_mono(original)
    watermarked = _to_mono(watermarked)
    n = min(len(original), len(watermarked))
    o = original[:n]
    w = watermarked[:n]
    noise = w - o
    ps = np.sum(o ** 2) + 1e-12
    pn = np.sum(noise ** 2) + 1e-12
    return float(10.0 * np.log10(ps / pn))


def random_payload(n_bits: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, size=n_bits)

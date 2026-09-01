> **⚠️ Proprietary — All Rights Reserved.** © 2026 Sandeep Grover. This repository is licensed to Sandeep Grover and may **not** be used, run, copied, modified, distributed, or used to train models without prior written permission. Public visibility does not grant a license. See [LICENSE](LICENSE).

---

# Blind Spread-Spectrum Audio Watermarking — Robustness & Detection Demo

A compact, **runnable** reference implementation of a blind audio watermark with a
detector built to **survive real-world transformations**, plus an honest robustness
harness that measures where it holds and where it breaks. Python / NumPy / SciPy /
FastAPI, WAV-first — no ML, no black boxes, every number reproducible from `src/harness.py`.

This is a capability demo for an audio-forensics / robustness engagement, not a product.

## The problem this addresses

Detection has to survive the channel: MP3 transcoding, resampling, requantization,
gain changes, EQ, additive noise, and time scaling — while the mark stays inaudible
and the false-positive rate stays controlled. Those goals trade against each other.
This demo makes the trade-offs measurable.

## Method (why it is robust)

* **Carrier:** mid-band (2–7 kHz) STFT magnitudes. High enough to be masked by
  program material, low enough to survive lossy codecs and resampling.
* **Spreading:** a secret key seeds a ±1 chip field; each payload bit is spread
  across thousands of time-frequency coefficients (processing gain → noise immunity).
* **Perceptual shaping:** embedding is *multiplicative*, `|X'| = |X|·(1 + α·c·b)`,
  so watermark energy tracks the local host magnitude and stays under the masking
  threshold (~41 dB SNR here).
* **Blind detection:** the host spectral envelope is estimated by a short
  frequency-axis moving average and removed (the chip field is white in the frequency
  index, so it survives that whitening). The residual correlates with the key chips →
  per-bit sign recovers the payload, and the aggregate matched energy is a
  **payload-blind present/absent statistic** for ROC / false-positive control.
* **Resync detector:** time scaling desyncs the analysis frames — the classic reason
  naive detectors fail in the field. `detect_resync()` searches a small set of
  resampling hypotheses and keeps the strongest, recovering the payload.

## Measured robustness (40 clips, 16-bit payload, α=0.35)

Reproduce with `python3 src/harness.py`. Verbatim from `examples/robustness_report.json`:

| Transformation      | mean BER | detection rate @ 1% FPR |
|---------------------|:--------:|:-----------------------:|
| none (clean)        |  0.000   |         100%            |
| MP3 128 kbps        |  0.000   |         100%            |
| MP3 64 kbps         |  0.000   |         100%            |
| resample 44.1→22→44 |  0.000   |         100%            |
| additive noise 30 dB|  0.002   |         100%            |
| additive noise 25 dB|  0.034   |         100%            |
| additive noise 20 dB|  0.188   |         7.5%  ← breaks   |
| requantize 8-bit    |  0.000   |         100%            |
| gain ×0.5           |  0.000   |         100%            |
| low-pass 8 kHz      |  0.000   |         100%            |
| high-pass 150 Hz    |  0.000   |         100%            |
| time-stretch +2% (resync) | 0.000 |       100%            |

Audibility: **mean 41 dB SNR**. Clean BER: **0**. Present-vs-absent **ROC AUC: 1.0**.

**Honest boundaries** (this is the point of a robustness report):
* Additive noise at **20 dB SNR** — audible hiss — degrades the payload (BER ≈ 0.19)
  and drops below the detection threshold. 25–30 dB and above is solid. Fixing 20 dB
  cleanly needs error-correction coding + a shorter payload, a design choice.
* Time-stretch is only recovered **with the resync detector**; the naive detector
  fails (BER ≈ 0.64 → 0.00 with resync). Larger scale factors need a finer or wider
  rate search, or a self-synchronizing pilot sequence.
* No embedding-side psychoacoustic model beyond magnitude proportionality; a real
  system would add a MPEG-style masking threshold and per-frame gain control.

## Run it

```bash
pip install -r requirements.txt

python3 src/cli.py demo                 # embed → blind-detect a payload, self-contained
python3 src/harness.py                  # full robustness table (synthetic clips)
python3 src/harness.py your_audio.wav   # ...on your own WAV
python3 -m pytest tests -q              # correctness tests

uvicorn api:app --app-dir src           # REST service
#   POST /embed  (wav + payload_hex + key)  -> watermarked wav (X-Payload-Hex header)
#   POST /detect (wav + key [+ resync])     -> {watermark_present, statistic, payload_hex}
```

## Layout

```
src/watermark.py   embed / detect / detect_resync / metrics  (the engine)
src/attacks.py     MP3, resample, noise sweep, requantize, gain, EQ, time-stretch
src/harness.py     population robustness + ROC/AUC + threshold @ target FPR
src/api.py         FastAPI service (/embed, /detect, /health)
src/cli.py         command-line demo
tests/             correctness tests (clean, absent/wrong-key, MP3, resync)
```

Built as a working sample by Dr. Sandeep Grover.

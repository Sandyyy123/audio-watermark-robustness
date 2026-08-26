"""Minimal correctness tests. Run: python3 -m pytest tests -q  (or python3 tests/test_watermark.py)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from watermark import WatermarkConfig, embed, detect, detect_resync, ber, snr_db, random_payload
from audio_io import synth_signal
from attacks import mp3_roundtrip, time_stretch, additive_noise

KEY = 20260826


def test_clean_roundtrip_is_lossless_and_inaudible():
    cfg = WatermarkConfig()
    x = synth_signal(cfg.sr, 3.0)
    pay = random_payload(cfg.payload_bits, seed=1)
    y = embed(x, pay, cfg, KEY)
    r = detect(y, cfg, KEY)
    assert ber(pay, r["bits"]) == 0.0
    assert snr_db(x, y) > 30.0            # inaudible headroom
    assert r["statistic"] > 15.0


def test_absent_and_wrong_key_are_rejected():
    cfg = WatermarkConfig()
    x = synth_signal(cfg.sr, 3.0)
    pay = random_payload(cfg.payload_bits, seed=1)
    y = embed(x, pay, cfg, KEY)
    assert detect(x, cfg, KEY)["statistic"] < detect(y, cfg, KEY)["statistic"] / 3
    assert detect(y, cfg, KEY + 1)["statistic"] < detect(y, cfg, KEY)["statistic"] / 3


def test_survives_mp3():
    cfg = WatermarkConfig()
    x = synth_signal(cfg.sr, 3.0)
    pay = random_payload(cfg.payload_bits, seed=2)
    y = embed(x, pay, cfg, KEY)
    r = detect(mp3_roundtrip(y, cfg.sr), cfg, KEY)
    assert ber(pay, r["bits"]) < 0.05


def test_resync_recovers_time_stretch():
    cfg = WatermarkConfig()
    x = synth_signal(cfg.sr, 3.0)
    pay = random_payload(cfg.payload_bits, seed=3)
    y = embed(x, pay, cfg, KEY)
    ya = time_stretch(y, cfg.sr, 1.02)
    assert ber(pay, detect(ya, cfg, KEY)["bits"]) > 0.2          # naive fails
    assert ber(pay, detect_resync(ya, cfg, KEY)["bits"]) < 0.05  # resync recovers


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"PASS {name}")
    print("all tests passed")

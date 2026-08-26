"""
Command-line demo: embed a watermark into a WAV, run the robustness suite, or
detect a payload.

  python3 src/cli.py demo                       # end-to-end self-contained demo
  python3 src/cli.py embed in.wav out.wav 1a2b  # embed 16-bit hex payload
  python3 src/cli.py detect out.wav             # blind detect
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from watermark import WatermarkConfig, embed, detect, detect_resync, ber, snr_db, random_payload
from audio_io import read_wav, write_wav, synth_signal


def _hex_to_bits(h, n):
    v = int(h, 16)
    return np.array([(v >> (n - 1 - i)) & 1 for i in range(n)], dtype=int)


def _bits_to_hex(bits):
    return hex(int("".join(str(int(b)) for b in bits), 2))[2:].zfill(len(bits) // 4)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    cfg = WatermarkConfig()
    KEY = 20260826
    if cmd == "demo":
        x = synth_signal(cfg.sr, 4.0)
        write_wav("examples/host.wav", x, cfg.sr)
        payload = _hex_to_bits("beef", cfg.payload_bits)
        y = embed(x, payload, cfg, KEY)
        write_wav("examples/watermarked.wav", y, cfg.sr)
        print(f"payload embedded: 0x{_bits_to_hex(payload)}   SNR={snr_db(x, y):.1f} dB")
        r = detect(y, cfg, KEY)
        print(f"blind detect: 0x{_bits_to_hex(r['bits'])}  stat={r['statistic']:.1f}  BER={ber(payload, r['bits']):.3f}")
        print("wrote examples/host.wav, examples/watermarked.wav")
    elif cmd == "embed":
        inp, outp = sys.argv[2], sys.argv[3]
        payload = _hex_to_bits(sys.argv[4], cfg.payload_bits) if len(sys.argv) > 4 else random_payload(cfg.payload_bits)
        x, sr = read_wav(inp)
        cfg = WatermarkConfig(sr=sr)
        y = embed(x, payload, cfg, KEY)
        write_wav(outp, y, sr)
        print(f"embedded 0x{_bits_to_hex(payload)} into {outp}  (SNR={snr_db(x, y):.1f} dB)")
    elif cmd == "detect":
        x, sr = read_wav(sys.argv[2])
        cfg = WatermarkConfig(sr=sr)
        r = detect_resync(x, cfg, KEY)
        present = r["statistic"] > 4.94
        print(f"present={present}  stat={r['statistic']:.1f}  payload=0x{_bits_to_hex(r['bits']) if present else None}  rate={r.get('rate',1.0)}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()

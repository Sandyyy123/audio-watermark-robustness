"""
Robustness + detection-reliability evaluation harness.

Produces, over a population of clips:
  * embedding SNR (audibility proxy)
  * clean bit-error rate
  * ROC AUC and a threshold set for a target false-positive rate
  * per-attack: mean BER and detection rate (payload-blind) at that threshold

Run:  python3 src/harness.py            (synthetic clips, self-contained)
      python3 src/harness.py path.wav   (use a real WAV as the host)
"""
from __future__ import annotations
import json
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from watermark import WatermarkConfig, embed, detect, detect_resync, ber, snr_db, random_payload
from audio_io import synth_signal, read_wav
from attacks import attack_suite

KEY = 20260826
N_CLIPS = 40
TARGET_FPR = 0.01


def _roc_auc(scores_pos, scores_neg) -> float:
    pos = np.asarray(scores_pos)
    neg = np.asarray(scores_neg)
    n = 0
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
        n += len(neg)
    return float(wins / n) if n else float("nan")


def _threshold_at_fpr(scores_neg, fpr: float) -> float:
    neg = np.sort(np.asarray(scores_neg))[::-1]
    k = max(0, int(np.ceil(fpr * len(neg))) - 1)
    return float(neg[k])


def make_clips(host_path=None, cfg=None):
    cfg = cfg or WatermarkConfig()
    clips = []
    if host_path:
        x, sr = read_wav(host_path)
        if x.ndim == 2:
            x = x.mean(axis=1)
        cfg.sr = sr
        seg = int(cfg.sr * 2.0)
        for i in range(0, len(x) - seg, seg):
            clips.append(x[i:i + seg])
            if len(clips) >= N_CLIPS:
                break
        if not clips:
            clips = [x]
    else:
        for s in range(N_CLIPS):
            clips.append(synth_signal(cfg.sr, 2.0, seed=100 + s))
    return clips, cfg


def run(host_path=None):
    clips, cfg = make_clips(host_path)
    attacks = attack_suite()
    payload = random_payload(cfg.payload_bits, seed=42)

    snrs, clean_ber = [], []
    wm_stat_clean, host_stat_clean = [], []      # for ROC
    per_attack = {a: {"ber": [], "stat": []} for a in attacks}

    for x in clips:
        y = embed(x, payload, cfg, KEY)
        snrs.append(snr_db(x, y))
        # ROC population (attack = none): watermarked vs host-only
        wm_stat_clean.append(detect(y, cfg, KEY)["statistic"])
        host_stat_clean.append(detect(x, cfg, KEY)["statistic"])
        r0 = detect(y, cfg, KEY)
        clean_ber.append(ber(payload, r0["bits"]))
        for name, fn in attacks.items():
            ya = fn(y, cfg.sr)
            # time scaling desyncs frames -> use the resync-search detector
            res = detect_resync(ya, cfg, KEY) if "stretch" in name else detect(ya, cfg, KEY)
            per_attack[name]["ber"].append(ber(payload, res["bits"]))
            per_attack[name]["stat"].append(res["statistic"])

    auc = _roc_auc(wm_stat_clean, host_stat_clean)
    thr = _threshold_at_fpr(host_stat_clean, TARGET_FPR)

    summary = {
        "config": {
            "n_fft": cfg.n_fft, "hop": cfg.hop, "band_hz": [cfg.band_lo_hz, cfg.band_hi_hz],
            "alpha": cfg.alpha, "payload_bits": cfg.payload_bits, "sr": cfg.sr,
            "n_clips": len(clips), "target_fpr": TARGET_FPR,
        },
        "audibility_snr_db_mean": round(float(np.mean(snrs)), 2),
        "clean_ber_mean": round(float(np.mean(clean_ber)), 5),
        "roc_auc_present_vs_absent": round(auc, 4),
        "detection_threshold_at_fpr": round(thr, 3),
        "attacks": {},
    }
    for name in attacks:
        bers = np.array(per_attack[name]["ber"])
        stats = np.array(per_attack[name]["stat"])
        summary["attacks"][name] = {
            "mean_ber": round(float(bers.mean()), 4),
            "detection_rate": round(float(np.mean(stats > thr)), 4),
            "mean_statistic": round(float(stats.mean()), 2),
        }
    return summary


def _print_table(s):
    print("\n=== Blind Spread-Spectrum Audio Watermark - Robustness Report ===")
    print(f"clips={s['config']['n_clips']}  payload={s['config']['payload_bits']} bits  "
          f"band={s['config']['band_hz'][0]:.0f}-{s['config']['band_hz'][1]:.0f}Hz  alpha={s['config']['alpha']}")
    print(f"Audibility (mean SNR): {s['audibility_snr_db_mean']} dB   "
          f"Clean BER: {s['clean_ber_mean']}   "
          f"ROC AUC: {s['roc_auc_present_vs_absent']}   "
          f"threshold@FPR={s['config']['target_fpr']}: {s['detection_threshold_at_fpr']}")
    print(f"\n{'attack':<20}{'mean BER':>10}{'detect rate':>14}{'mean stat':>12}")
    print("-" * 56)
    for name, a in s["attacks"].items():
        print(f"{name:<20}{a['mean_ber']:>10.3f}{a['detection_rate']:>14.2%}{a['mean_statistic']:>12.1f}")


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else None
    s = run(host)
    _print_table(s)
    with open("examples/robustness_report.json", "w") as f:
        json.dump(s, f, indent=2)
    print("\nSaved examples/robustness_report.json")

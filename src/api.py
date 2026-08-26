"""
FastAPI service for blind audio watermarking.

  POST /embed   (multipart: wav file + payload_hex + key)  -> watermarked WAV
  POST /detect  (multipart: wav file + key)                -> payload + confidence
  GET  /health

Run:  uvicorn api:app --app-dir src --reload
"""
from __future__ import annotations
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from scipy.io import wavfile

from watermark import WatermarkConfig, embed, detect, detect_resync, random_payload

app = FastAPI(title="Blind Audio Watermark API", version="0.1.0")
CFG = WatermarkConfig()


def _read_upload(raw: bytes):
    sr, data = wavfile.read(io.BytesIO(raw))
    if data.dtype == np.int16:
        x = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float64) / 2147483648.0
    else:
        x = data.astype(np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x, sr


def _bits_to_hex(bits) -> str:
    b = "".join(str(int(v)) for v in bits)
    return hex(int(b, 2))[2:].zfill((len(b) + 3) // 4)


def _hex_to_bits(h: str, n: int):
    val = int(h, 16)
    return np.array([(val >> (n - 1 - i)) & 1 for i in range(n)], dtype=int)


@app.get("/health")
def health():
    return {"status": "ok", "payload_bits": CFG.payload_bits,
            "band_hz": [CFG.band_lo_hz, CFG.band_hi_hz]}


@app.post("/embed")
async def embed_ep(file: UploadFile = File(...),
                   payload_hex: str = Form(""),
                   key: int = Form(20260826)):
    x, sr = _read_upload(await file.read())
    cfg = WatermarkConfig(sr=sr)
    if payload_hex:
        try:
            payload = _hex_to_bits(payload_hex, cfg.payload_bits)
        except ValueError:
            raise HTTPException(400, "payload_hex not valid hex")
    else:
        payload = random_payload(cfg.payload_bits, seed=0)
    y = embed(x, payload, cfg, key)
    buf = io.BytesIO()
    wavfile.write(buf, sr, np.clip(y, -1, 1).astype(np.float32))
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="audio/wav",
        headers={"X-Payload-Hex": _bits_to_hex(payload),
                 "Content-Disposition": "attachment; filename=watermarked.wav"})


@app.post("/detect")
async def detect_ep(file: UploadFile = File(...),
                    key: int = Form(20260826),
                    resync: bool = Form(False)):
    x, sr = _read_upload(await file.read())
    cfg = WatermarkConfig(sr=sr)
    res = detect_resync(x, cfg, key) if resync else detect(x, cfg, key)
    present = bool(res["statistic"] > 4.94)   # threshold @ ~1% FPR (see harness)
    return JSONResponse({
        "watermark_present": present,
        "detection_statistic": round(res["statistic"], 3),
        "payload_hex": _bits_to_hex(res["bits"]) if present else None,
        "resync_rate": res.get("rate", 1.0),
    })

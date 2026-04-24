#!/usr/bin/env python3
"""
Local faster-whisper HTTP server -- compatible with stt_worker.py
Exposes POST /v1/transcribe on port 8766.

Request:  raw PCM float32 bytes, query params: sample_rate, language, initial_prompt
Response: {"text": "...", "language": "...", "processing_s": N}

Environment variables:
  WHISPER_MODEL_SIZE   Model to load (default: "small"). Use "large-v3" on GPU servers.
  WHISPER_DEVICE       Device: "cpu" or "cuda" (default: "cpu").
  WHISPER_COMPUTE_TYPE Quantization: "int8", "float16", "int8_float16" (default: "int8").
"""

import asyncio
import os
import time
import numpy as np
from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse
import uvicorn
from faster_whisper import WhisperModel
from log import logger
from prompts.utils import WHISPER_DEFAULT_PROMPT

MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")
DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

# Default language when the client sends "" or "auto".
# Set to "" to keep auto-detection.
DEFAULT_LANGUAGE = "fr"

logger.info(f"Loading Whisper model '{MODEL_SIZE}' on {DEVICE} ({COMPUTE_TYPE})...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
logger.info("Model ready.")

app = FastAPI()


@app.post("/v1/transcribe")
async def transcribe(
    request: Request,
    sample_rate: int = Query(16000),
    language: str = Query(""),
    initial_prompt: str = Query(""),
):
    raw = await request.body()
    if not raw:
        return JSONResponse({"text": "", "language": "", "processing_s": 0})

    audio = np.frombuffer(raw, dtype=np.float32).copy()
    lang = language if language and language != "auto" else DEFAULT_LANGUAGE
    prompt = initial_prompt or WHISPER_DEFAULT_PROMPT

    t0 = time.time()

    def _run():
        segs, info = model.transcribe(
            audio,
            language=lang,
            initial_prompt=prompt,
            beam_size=5,
            vad_filter=True,
            temperature=0,
            condition_on_previous_text=True,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
        )
        return list(segs), info

    segments, info = await asyncio.to_thread(_run)
    text = " ".join(s.text for s in segments).strip()
    processing_s = time.time() - t0

    return JSONResponse({
        "text": text,
        "language": info.language,
        "processing_s": round(processing_s, 2),
    })


if __name__ == "__main__":
    logger.info(f"Whisper server starting on http://localhost:8766 (model={MODEL_SIZE}, device={DEVICE})")
    uvicorn.run(app, host="0.0.0.0", port=8766, log_level="warning")

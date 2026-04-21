#!/usr/bin/env python3
"""
Local faster-whisper HTTP server -- compatible with stt_worker.py
Exposes POST /v1/transcribe on port 8766.

Request:  raw PCM float32 bytes, query params: sample_rate, language, initial_prompt
Response: {"text": "...", "language": "...", "processing_s": N}
"""

import time
import numpy as np
from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse
import uvicorn
from faster_whisper import WhisperModel

MODEL_SIZE = "small"

# Default language when the client sends "" or "auto".
# Set to "" to keep auto-detection.
DEFAULT_LANGUAGE = "fr"

# Domain-specific prompt injected when the client sends no initial_prompt.
# Helps Whisper handle technical vocabulary and proper nouns correctly.
DEFAULT_PROMPT = (
    "Discussion en français sur l'intelligence artificielle, "
    "les modèles de langage, la transcription vocale, les graphes de connaissances, "
    "Mímir, Whisper, Anthropic, Azure, Ollama, FastAPI."
)

print(f"  Loading Whisper model '{MODEL_SIZE}' on CPU (int8)...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print(f"  Model ready.")

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
    prompt = initial_prompt or DEFAULT_PROMPT

    t0 = time.time()
    segments, info = model.transcribe(
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
    text = " ".join(s.text for s in segments).strip()
    processing_s = time.time() - t0

    return JSONResponse({
        "text": text,
        "language": info.language,
        "processing_s": round(processing_s, 2),
    })


if __name__ == "__main__":
    print("  Whisper server starting on http://localhost:8766")
    uvicorn.run(app, host="0.0.0.0", port=8766, log_level="warning")

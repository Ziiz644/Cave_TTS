import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import torchaudio as ta

from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # multilingual

app = FastAPI()

MODEL = None
SR = 24000  # Chatterbox uses 24k in examples; we'll export wav at model.sr

# Simple local voice registry (you can later move this to DB)

VOICE_MAP = {
    "basma": "voices/basma.wav",
    "layla": "voices/sirin.wav",
}


class TtsRequest(BaseModel):
    text: str
    language_id: str = "ar"     # "ar" for Arabic, "en" for English, etc.
    voice_id: str | None = None # optional - picks a reference clip if present

@app.on_event("startup")
def load_model():
    global MODEL
    device = "cpu"  # Render is CPU unless you pay for GPU
    MODEL = ChatterboxMultilingualTTS.from_pretrained(device=device)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/tts")
def tts(req: TtsRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    audio_prompt_path = None
    if req.voice_id:
        audio_prompt_path = ensure_wav(VOICE_MAP.get(req.voice_id))
        if not audio_prompt_path or not os.path.exists(audio_prompt_path):
            raise HTTPException(status_code=404, detail=f"Unknown voice_id: {req.voice_id}")

    # Generate
    wav = MODEL.generate(
        req.text,
        language_id=req.language_id,
        audio_prompt_path=audio_prompt_path
    )

    # Encode to WAV bytes
    out_path = "/tmp/out.wav"
    ta.save(out_path, wav, MODEL.sr)

    with open(out_path, "rb") as f:
        wav_bytes = f.read()

    return Response(content=wav_bytes, media_type="audio/wav")

@app.get("/voices")
def list_voices():
    return {
        "voices": list(VOICE_MAP.keys())
    }

import subprocess
from pathlib import Path

def ensure_wav(path: str) -> str:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Voice file not found: {p}")

    if p.suffix.lower() == ".wav":
        return str(p)

    out = p.with_suffix(".wav")

    if out.exists():
        return str(out)

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(p),
            "-ac", "1",
            "-ar", "24000",
            "-sample_fmt", "s16",
            str(out),
        ],
        check=True
    )

    return str(out)

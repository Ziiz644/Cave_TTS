from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
import os, uuid
from pathlib import Path

import torch
import torchaudio as ta

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

app = FastAPI()

OUT_DIR = Path("/tmp/tts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
MODEL = None

@app.on_event("startup")
def load_model():
    global MODEL
  
    MODEL = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
    # Optional: speed/VRAM improvements when using GPU
    if DEVICE == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE}

@app.post("/api/tts/speak")
async def speak(
    text: str = Form(...),
    language_id: str = Form("ar"),          # "ar" or "en"
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5),
    voice: UploadFile | None = File(None),  # optional voice cloning reference
):
    if not text.strip():
        return JSONResponse({"error": "text is required"}, status_code=400)

    # Save optional reference audio (voice cloning)
    audio_prompt_path = None
    if voice is not None:
        ref_path = OUT_DIR / f"ref_{uuid.uuid4().hex}_{voice.filename}"
        ref_path.write_bytes(await voice.read())
        audio_prompt_path = str(ref_path)

    out_wav = OUT_DIR / f"{uuid.uuid4().hex}.wav"

    # Generate
    wav = MODEL.generate(
        text,
        language_id=language_id,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )

    ta.save(str(out_wav), wav, MODEL.sr)
    return FileResponse(str(out_wav), media_type="audio/wav", filename="speech.wav")

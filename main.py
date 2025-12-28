from fastapi import FastAPI
from fastapi.responses import FileResponse
import subprocess, uuid
from pathlib import Path

app = FastAPI()

OUT_DIR = Path("/tmp/tts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PIPER_BIN = "piper"
MODEL_PATH = "models/ar.onnx"
CONFIG_PATH = "models/ar.onnx.json"

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/tts/speak")
def speak(payload: dict):
    text = (payload.get("text") or "").strip()
    if not text:
        return {"error": "text is required"}

    out_wav = OUT_DIR / f"{uuid.uuid4().hex}.wav"

    proc = subprocess.run(
        [PIPER_BIN, "-m", MODEL_PATH, "-c", CONFIG_PATH, "-f", str(out_wav)],
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if proc.returncode != 0:
        return {"error": "tts_failed", "details": proc.stderr.decode("utf-8", "ignore")}

    return FileResponse(str(out_wav), media_type="audio/wav", filename="speech.wav")

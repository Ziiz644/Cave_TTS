from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import subprocess, uuid, os
from pathlib import Path

app = FastAPI()

OUT_DIR = Path("/tmp/tts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# We'll install Piper CLI inside the image, so "piper" exists
PIPER_BIN = "piper"

# Where voice files live inside the container
VOICES_DIR = Path(os.getenv("VOICES_DIR", "/app/voices"))

class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice_id: str = Field(..., description="Relative path under /app/voices without extension, e.g. 'en/en_US-amy-low'")
    speaker_id: int | None = Field(default=None, description="For multi-speaker models only")
    length_scale: float | None = Field(default=None, description="Speaking speed/cadence (model-dependent)")
    noise_scale: float | None = Field(default=None, description="Variation (model-dependent)")
    noise_w: float | None = Field(default=None, description="Variation (model-dependent)")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/api/tts/voices")
def list_voices():
    # List all *.onnx in VOICES_DIR (excluding nested configs)
    voices = []
    for onnx in VOICES_DIR.rglob("*.onnx"):
        rel = onnx.relative_to(VOICES_DIR).as_posix()
        # voice_id is path without ".onnx"
        voices.append(rel[:-5])
    voices.sort()
    return {"voices": voices}

@app.post("/api/tts/speak")
def speak(req: SpeakRequest):
    text = req.text.strip()
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)

    model_path = VOICES_DIR / f"{req.voice_id}.onnx"
    config_path = VOICES_DIR / f"{req.voice_id}.onnx.json"

    if not model_path.exists() or not config_path.exists():
        return JSONResponse(
            {"error": "unknown_voice", "details": f"Missing {model_path} or {config_path}"},
            status_code=404,
        )

    out_wav = OUT_DIR / f"{uuid.uuid4().hex}.wav"

    cmd = [
        PIPER_BIN,
        "-m", str(model_path),
        "-c", str(config_path),
        "-f", str(out_wav),
    ]

    # Optional parameters (supported by Piper builds; safe to pass only when provided)
    if req.speaker_id is not None:
        cmd += ["-s", str(req.speaker_id)]  # speaker_id supported in Piper CLI :contentReference[oaicite:3]{index=3}
    if req.length_scale is not None:
        cmd += ["--length_scale", str(req.length_scale)]
    if req.noise_scale is not None:
        cmd += ["--noise_scale", str(req.noise_scale)]
    if req.noise_w is not None:
        cmd += ["--noise_w", str(req.noise_w)]

    proc = subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if proc.returncode != 0:
        return JSONResponse(
            {"error": "tts_failed", "details": proc.stderr.decode("utf-8", "ignore")},
            status_code=500,
        )

    return FileResponse(str(out_wav), media_type="audio/wav", filename="speech.wav")

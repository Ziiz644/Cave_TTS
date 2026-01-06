import os
import uuid
import re
import tempfile
from pathlib import Path
from typing import Optional, Generator

import numpy as np
import httpx
import torch
import torchaudio as ta

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

APP_NAME = "chatterbox-tts"

# ---------------- Config (ENV) ----------------
USE_GPU = os.getenv("USE_GPU", "true").lower() in ("1", "true", "yes", "y")
DEFAULT_LANG = os.getenv("DEFAULT_LANG", "ar").strip().lower()  # "ar" or "en"
ALLOW_SPEAKER_URL = os.getenv("ALLOW_SPEAKER_URL", "true").lower() in ("1", "true", "yes", "y")

MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "2000"))
OUTPUT_SAMPLE_RATE = int(os.getenv("OUTPUT_SAMPLE_RATE", "24000"))  # for streaming PCM output
MAX_SPEAKER_WAV_BYTES = int(os.getenv("MAX_SPEAKER_WAV_BYTES", str(8 * 1024 * 1024)))
SPEAKER_HTTP_TIMEOUT = float(os.getenv("SPEAKER_HTTP_TIMEOUT", "20"))

MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "220"))
MAX_CONCURRENT_SYNTH = int(os.getenv("MAX_CONCURRENT_SYNTH", "1"))

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8100,http://127.0.0.1:8100,https://api.mda.sa,https://mda.sa,https://www.mda.sa",
).split(",")

OUT_DIR = Path(os.getenv("OUT_DIR", "/tmp/tts"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------

app = FastAPI(title=APP_NAME, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "X-Request-Id", "X-Audio-Format", "X-Sample-Rate"],
)

DEVICE = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
MODEL: Optional[ChatterboxMultilingualTTS] = None

try:
    import threading
    _sem = threading.BoundedSemaphore(MAX_CONCURRENT_SYNTH)
except Exception:
    _sem = None

_SENT_SPLIT = re.compile(r"(?<=[\.\!\?\u061F])\s+")


def _acquire_sem_or_503():
    if _sem is None:
        return
    ok = _sem.acquire(timeout=60)
    if not ok:
        raise HTTPException(status_code=503, detail="TTS server is busy, try again")


def _release_sem():
    if _sem is None:
        return
    try:
        _sem.release()
    except Exception:
        pass


def chunk_text(text: str, max_chars: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    out: list[str] = []
    buf = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def _float_to_s16le_pcm(audio: np.ndarray) -> bytes:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    return pcm16.tobytes(order="C")


async def _download_to_temp(url: str) -> str:
    if not ALLOW_SPEAKER_URL:
        raise HTTPException(status_code=400, detail="speaker_wav_url is disabled")
    if not url or not url.strip():
        raise HTTPException(status_code=400, detail="speaker_wav_url is empty")

    fd, tmp_path = tempfile.mkstemp(prefix="chatter_ref_", suffix=".wav")
    os.close(fd)

    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
    timeout = httpx.Timeout(SPEAKER_HTTP_TIMEOUT)
    total = 0

    try:
        async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:
            async with client.stream("GET", url.strip()) as resp:
                if resp.status_code >= 400:
                    raise HTTPException(status_code=400, detail=f"Failed to download speaker wav (HTTP {resp.status_code})")

                with open(tmp_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > MAX_SPEAKER_WAV_BYTES:
                            raise HTTPException(status_code=400, detail="speaker wav too large (max 8MB)")
                        f.write(chunk)

        if total == 0:
            raise HTTPException(status_code=400, detail="Downloaded speaker wav is empty")

        # Quick RIFF/WAVE signature check (avoid HTML/403 surprises)
        with open(tmp_path, "rb") as f:
            head = f.read(12)
        if len(head) < 12 or head[0:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise HTTPException(status_code=400, detail="speaker_wav_url did not return a valid WAV")

        return tmp_path

    except HTTPException:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise
    except Exception as e:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"Error downloading speaker wav: {str(e)}")


@app.on_event("startup")
def load_model():
    global MODEL
    MODEL = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
    if DEVICE == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


@app.get("/health")
def health():
    return {
        "service": APP_NAME,
        "ok": True,
        "device": DEVICE,
        "gpu_available": torch.cuda.is_available(),
        "default_lang": DEFAULT_LANG,
        "output_sample_rate": OUTPUT_SAMPLE_RATE,
        "max_text_chars": MAX_TEXT_CHARS,
        "max_concurrent_synth": MAX_CONCURRENT_SYNTH,
        "max_chunk_chars": MAX_CHUNK_CHARS,
    }


# ==========================================================
# 1) NORMAL endpoint (WAV) — for TTSLab testing + samples
# ==========================================================
@app.post("/api/tts/speak")
async def speak_form(
    text: str = Form(...),
    language_id: str = Form(DEFAULT_LANG),
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5),
    voice: UploadFile | None = File(None),
):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    text = (text or "").strip()
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)
    if len(text) > MAX_TEXT_CHARS:
        return JSONResponse({"error": f"text too long (max {MAX_TEXT_CHARS})"}, status_code=400)

    language_id = (language_id or DEFAULT_LANG).strip().lower()
    if language_id not in ("ar", "en"):
        return JSONResponse({"error": 'language_id must be "ar" or "en"'}, status_code=400)

    audio_prompt_path = None
    tmp_ref = None
    try:
        if voice is not None:
            ref_path = OUT_DIR / f"ref_{uuid.uuid4().hex}_{voice.filename}"
            ref_path.write_bytes(await voice.read())
            audio_prompt_path = str(ref_path)
            tmp_ref = audio_prompt_path

        out_wav = OUT_DIR / f"{uuid.uuid4().hex}.wav"

        _acquire_sem_or_503()
        wav = MODEL.generate(
            text,
            language_id=language_id,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )

        ta.save(str(out_wav), wav, MODEL.sr)
        return FileResponse(str(out_wav), media_type="audio/wav", filename="speech.wav")

    finally:
        _release_sem()
        if tmp_ref and os.path.exists(tmp_ref):
            try:
                os.remove(tmp_ref)
            except Exception:
                pass


# ==========================================================
# 2) STREAMING endpoint (PCM s16le) — for chat
#    Accepts speaker_wav_url (like XTTS) or speaker_wav_path
# ==========================================================
class StreamRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    language: str = Field(default=DEFAULT_LANG)

    speaker_wav_url: Optional[str] = None
    speaker_wav_path: Optional[str] = None

    # backward compat field name if you want:
    speaker_wav: Optional[str] = None

    exaggeration: float = Field(default=0.5)
    cfg_weight: float = Field(default=0.5)

    format: str = Field(default="pcm")  # only pcm supported here
    sample_rate: int = Field(default=OUTPUT_SAMPLE_RATE)

    @field_validator("language")
    @classmethod
    def validate_lang(cls, v: str) -> str:
        v = (v or DEFAULT_LANG).strip().lower()
        if v not in ("ar", "en"):
            raise ValueError('language must be "ar" or "en"')
        return v

    @field_validator("format")
    @classmethod
    def validate_fmt(cls, v: str) -> str:
        v = (v or "pcm").strip().lower()
        if v != "pcm":
            raise ValueError('format must be "pcm"')
        return v

    @model_validator(mode="after")
    def normalize_speaker_fields(self):
        if (not self.speaker_wav_url) and self.speaker_wav:
            self.speaker_wav_url = self.speaker_wav
        return self


@app.post("/api/tts/stream")
async def stream_pcm(request: Request, payload: StreamRequest = Body(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    req_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())

    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    if payload.sample_rate != OUTPUT_SAMPLE_RATE:
        raise HTTPException(status_code=400, detail=f"sample_rate must be {OUTPUT_SAMPLE_RATE}")

    language_id = payload.language
    exaggeration = float(payload.exaggeration)
    cfg_weight = float(payload.cfg_weight)

    tmp_ref = None
    try:
        # Resolve ref audio
        audio_prompt_path = None
        if payload.speaker_wav_url:
            tmp_ref = await _download_to_temp(payload.speaker_wav_url.strip())
            audio_prompt_path = tmp_ref
        elif payload.speaker_wav_path:
            p = payload.speaker_wav_path.strip()
            if not os.path.exists(p):
                raise HTTPException(status_code=400, detail=f"speaker_wav_path not found: {p}")
            audio_prompt_path = p

        chunks = chunk_text(text, MAX_CHUNK_CHARS)

        def gen() -> Generator[bytes, None, None]:
            _acquire_sem_or_503()
            try:
                for ch in chunks:
                    wav = MODEL.generate(
                        ch,
                        language_id=language_id,
                        audio_prompt_path=audio_prompt_path,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                    )
                    # resample to OUTPUT_SAMPLE_RATE if needed
                    # MODEL.sr might be different; enforce OUTPUT_SAMPLE_RATE for your frontend
                    if int(MODEL.sr) != int(OUTPUT_SAMPLE_RATE):
                        wav = ta.functional.resample(wav, orig_freq=MODEL.sr, new_freq=OUTPUT_SAMPLE_RATE)

                    yield _float_to_s16le_pcm(wav.cpu().numpy() if hasattr(wav, "cpu") else np.asarray(wav))
            finally:
                _release_sem()

        headers = {
            "X-Request-Id": req_id,
            "X-Audio-Format": "pcm",
            "X-Sample-Rate": str(OUTPUT_SAMPLE_RATE),
        }
        return StreamingResponse(gen(), media_type="application/octet-stream", headers=headers)

    finally:
        if tmp_ref and os.path.exists(tmp_ref):
            try:
                os.remove(tmp_ref)
            except Exception:
                pass

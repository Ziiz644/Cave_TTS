from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from TTS.api import TTS
import tempfile
import os

app = FastAPI(title="MDA Coqui TTS")

# Good Arabic option (multilingual):
# Runs on CPU fine, but first load is heavy.
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

class TtsReq(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/tts")
def synth(req: TtsReq):
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    # Synthesize to file
    tts.tts_to_file(text=req.text, file_path=path)

    # Return wav as a downloadable stream
    return FileResponse(path, media_type="audio/wav", filename="speech.wav")

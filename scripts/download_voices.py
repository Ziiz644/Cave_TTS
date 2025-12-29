import os, json, urllib.request
from pathlib import Path

VOICES_DIR = Path(os.getenv("VOICES_DIR", "/app/voices"))
VOICES_DIR.mkdir(parents=True, exist_ok=True)

# Keep this list SMALL for a lightweight Render build.
# Add more voices as needed.
SELECTED = [
    # Example English (United States) Amy low
    {
        "id": "en/en_US-amy-low",
        "model_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/low/en_US-amy-low.onnx?download=true",
        "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/low/en_US-amy-low.onnx.json?download=true",
    },

    # TODO: Add your Arabic voice URLs here (see notes below)
]

def download(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"✓ exists {path}")
        return
    print(f"↓ {url} -> {path}")
    urllib.request.urlretrieve(url, path)

def main():
    for v in SELECTED:
        voice_id = v["id"]
        model_path  = VOICES_DIR / f"{voice_id}.onnx"
        config_path = VOICES_DIR / f"{voice_id}.onnx.json"

        download(v["model_url"], model_path)
        download(v["config_url"], config_path)

    print("Done. Voices installed:")
    for onnx in VOICES_DIR.rglob("*.onnx"):
        print(" -", onnx.relative_to(VOICES_DIR))

if __name__ == "__main__":
    main()

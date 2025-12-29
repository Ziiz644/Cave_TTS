import os, json, urllib.request
from pathlib import Path

VOICES_DIR = Path(os.getenv("VOICES_DIR", "/app/voices"))
VOICES_DIR.mkdir(parents=True, exist_ok=True)

# Keep this list SMALL for a lightweight Render build.
# Add more voices as needed.
SELECTED = [

# =========================
# 🇺🇸 ENGLISH (16 voices)
# =========================

# US
{"id":"en/en_US-amy-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/low/en_US-amy-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/low/en_US-amy-low.onnx.json?download=true"},

{"id":"en/en_US-joe-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/joe/low/en_US-joe-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/joe/low/en_US-joe-low.onnx.json?download=true"},

{"id":"en/en_US-kristin-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/kristin/low/en_US-kristin-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/kristin/low/en_US-kristin-low.onnx.json?download=true"},

{"id":"en/en_US-ryan-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/low/en_US-ryan-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/low/en_US-ryan-low.onnx.json?download=true"},

# UK
{"id":"en/en_GB-alan-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/low/en_GB-alan-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/low/en_GB-alan-low.onnx.json?download=true"},

{"id":"en/en_GB-sarah-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/sarah/low/en_GB-sarah-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/sarah/low/en_GB-sarah-low.onnx.json?download=true"},

# Australia
{"id":"en/en_AU-nat-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_AU/nat/low/en_AU-nat-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_AU/nat/low/en_AU-nat-low.onnx.json?download=true"},

# Neutral / other English
{"id":"en/en_US-danny-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/danny/low/en_US-danny-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/danny/low/en_US-danny-low.onnx.json?download=true"},

{"id":"en/en_US-jenny-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/jenny/low/en_US-jenny-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/jenny/low/en_US-jenny-low.onnx.json?download=true"},

{"id":"en/en_US-bryce-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/bryce/low/en_US-bryce-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/bryce/low/en_US-bryce-low.onnx.json?download=true"},

{"id":"en/en_US-erin-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/erin/low/en_US-erin-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/erin/low/en_US-erin-low.onnx.json?download=true"},

{"id":"en/en_US-norman-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/norman/low/en_US-norman-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/norman/low/en_US-norman-low.onnx.json?download=true"},

{"id":"en/en_US-tim-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/tim/low/en_US-tim-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/tim/low/en_US-tim-low.onnx.json?download=true"},

{"id":"en/en_US-mary-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/mary/low/en_US-mary-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/mary/low/en_US-mary-low.onnx.json?download=true"},

{"id":"en/en_US-robert-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/robert/low/en_US-robert-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/robert/low/en_US-robert-low.onnx.json?download=true"},

{"id":"en/en_US-samantha-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/samantha/low/en_US-samantha-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/samantha/low/en_US-samantha-low.onnx.json?download=true"},

# =========================
# 🇸🇦 ARABIC (14 voices)
# =========================

{"id":"ar/ar-msa-1-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/msa/low/ar-msa-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/msa/low/ar-msa-low.onnx.json?download=true"},

{"id":"ar/ar-msa-2-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/msa2/low/ar-msa2-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/msa2/low/ar-msa2-low.onnx.json?download=true"},

{"id":"ar/ar-female-1-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/female/low/ar-female-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/female/low/ar-female-low.onnx.json?download=true"},

{"id":"ar/ar-male-1-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/male/low/ar-male-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/male/low/ar-male-low.onnx.json?download=true"},

{"id":"ar/ar-gulf-1-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/gulf/low/ar-gulf-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/gulf/low/ar-gulf-low.onnx.json?download=true"},

{"id":"ar/ar-gulf-2-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/gulf2/low/ar-gulf2-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/gulf2/low/ar-gulf2-low.onnx.json?download=true"},

{"id":"ar/ar-classical-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/classical/low/ar-classical-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/classical/low/ar-classical-low.onnx.json?download=true"},

{"id":"ar/ar-news-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/news/low/ar-news-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/news/low/ar-news-low.onnx.json?download=true"},

{"id":"ar/ar-quranic-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/quranic/low/ar-quranic-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/quranic/low/ar-quranic-low.onnx.json?download=true"},

{"id":"ar/ar-formal-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/formal/low/ar-formal-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/formal/low/ar-formal-low.onnx.json?download=true"},

{"id":"ar/ar-modern-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/modern/low/ar-modern-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/modern/low/ar-modern-low.onnx.json?download=true"},

{"id":"ar/ar-neutral-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/neutral/low/ar-neutral-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/neutral/low/ar-neutral-low.onnx.json?download=true"},

{"id":"ar/ar-female-2-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/female2/low/ar-female2-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/female2/low/ar-female2-low.onnx.json?download=true"},

{"id":"ar/ar-male-2-low",
 "model_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/male2/low/ar-male2-low.onnx?download=true",
 "config_url":"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/male2/low/ar-male2-low.onnx.json?download=true"},
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

# /app/scripts/download_voices.py
import os
import random
import urllib.request
import urllib.error
from pathlib import Path

VOICES_DIR = Path(os.environ.get("VOICES_DIR", "/app/voices"))
VOICES_DIR.mkdir(parents=True, exist_ok=True)

# ✅ A safe, known-good set (mostly English). Arabic availability is limited in Piper.
# You can expand this list gradually once your pipeline is stable.
CANDIDATES = [
    # ---------- English (US/GB/AU) ----------
    ("en/en_US-amy-low",
     "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/low/en_US-amy-low.onnx?download=true",
     "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/low/en_US-amy-low.onnx.json?download=true"),

    ("en/en_US-joe-low",
     "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/joe/low/en_US-joe-low.onnx?download=true",
     "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/joe/low/en_US-joe-low.onnx.json?download=true"),

    ("en/en_US-lessac-medium",
     "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx?download=true",
     "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json?download=true"),

    ("en/en_GB-alan-low",
     "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/low/en_GB-alan-low.onnx?download=true",
     "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/low/en_GB-alan-low.onnx.json?download=true"),

    ("en/en_GB-sarah-low",
     "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/sarah/low/en_GB-sarah-low.onnx?download=true",
     "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/sarah/low/en_GB-sarah-low.onnx.json?download=true"),

    ("en/en_AU-nat-low",
     "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_AU/nat/low/en_AU-nat-low.onnx?download=true",
     "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_AU/nat/low/en_AU-nat-low.onnx.json?download=true"),

    # ---------- Arabic (MSA; limited set) ----------
    ("ar/ar-msa-low",
     "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/msa/low/ar-msa-low.onnx?download=true",
     "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ar/ar/msa/low/ar-msa-low.onnx.json?download=true"),
]

def _download(url: str, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        # basic sanity check (avoid saving HTML error pages)
        if len(data) < 50_000:
            print(f"❌ Too small / suspicious ({len(data)} bytes): {url}")
            return False
        dst.write_bytes(data)
        print(f"✅ Saved: {dst} ({dst.stat().st_size//1024} KB)")
        return True
    except urllib.error.HTTPError as e:
        print(f"❌ HTTP {e.code} for {url}")
        return False
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False

def main():
    # Request “30 voices” but only download what actually exists in CANDIDATES.
    # You can expand CANDIDATES later safely.
    target = int(os.environ.get("VOICE_COUNT", "30"))

    pool = CANDIDATES.copy()
    random.shuffle(pool)

    ok_models = 0
    for voice_id, model_url, cfg_url in pool[:target]:
        out_dir = VOICES_DIR / voice_id
        model_path = out_dir / f"{voice_id.replace('/', '-')}.onnx"
        cfg_path   = out_dir / f"{voice_id.replace('/', '-')}.onnx.json"

        print(f"\n=== {voice_id} ===")
        m_ok = _download(model_url, model_path)
        c_ok = _download(cfg_url, cfg_path)

        if m_ok and c_ok:
            ok_models += 1
        else:
            # cleanup partials
            if model_path.exists(): model_path.unlink()
            if cfg_path.exists(): cfg_path.unlink()

    print(f"\nDone. Downloaded {ok_models} voice(s) into {VOICES_DIR}")

    # ✅ only fail build if you literally have nothing
    if ok_models == 0:
        raise SystemExit("No voices downloaded. Check URLs/connectivity.")

if __name__ == "__main__":
    main()

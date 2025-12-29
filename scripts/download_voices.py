import os
import json
import random
import urllib.request
import urllib.error
from pathlib import Path

VOICES_DIR = Path(os.environ.get("VOICES_DIR", "/app/voices"))
VOICES_DIR.mkdir(parents=True, exist_ok=True)

VOICES_INDEX_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/voices.json?download=true"
BASE_RESOLVE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"

VOICE_COUNT = int(os.environ.get("VOICE_COUNT", "30"))

# Comma-separated families; default: English + Arabic
FAMILIES = [x.strip() for x in os.environ.get("VOICE_FAMILIES", "en,ar").split(",") if x.strip()]

def http_get_bytes(url: str, timeout=180) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()

def download_to(url: str, dst: Path, min_bytes: int = 1) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = http_get_bytes(url)
        if len(data) < min_bytes:
            print(f"❌ Too small ({len(data)} bytes): {url}")
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
    print(f"Fetching voices index: {VOICES_INDEX_URL}")
    voices_json = json.loads(http_get_bytes(VOICES_INDEX_URL).decode("utf-8"))

    # voices_json is a dict keyed by voice key (e.g., "en_US-amy-medium")
    # each value has fields like: language, quality, files, etc. :contentReference[oaicite:2]{index=2}
    candidates = []
    for voice_key, meta in voices_json.items():
        lang = meta.get("language", {})
        family = lang.get("family")
        if family in FAMILIES:
            candidates.append((voice_key, meta))

    if not candidates:
        raise SystemExit(f"No voice candidates for families={FAMILIES}. Check VOICE_FAMILIES.")

    random.shuffle(candidates)
    selected = candidates[:VOICE_COUNT]

    ok = 0
    for voice_key, meta in selected:
        files = meta.get("files", {})
        # We only need the ONNX model + its JSON config
        onnx_path = None
        json_path = None
        for fp in files.keys():
            if fp.endswith(".onnx"):
                onnx_path = fp
            elif fp.endswith(".onnx.json"):
                json_path = fp

        if not onnx_path or not json_path:
            print(f"❌ Skipping {voice_key}: missing .onnx or .onnx.json in files")
            continue

        onnx_url = BASE_RESOLVE + onnx_path + "?download=true"
        json_url = BASE_RESOLVE + json_path + "?download=true"

        # Save under /app/voices using the same relative paths
        onnx_dst = VOICES_DIR / onnx_path
        json_dst = VOICES_DIR / json_path

        # Use expected sizes if present (best validation)
        onnx_min = max(500_000, int(files.get(onnx_path, {}).get("size_bytes", 0)) // 10 or 500_000)
        json_min = 200  # configs are small (a few KB) :contentReference[oaicite:3]{index=3}

        print(f"\n=== {voice_key} ({meta.get('quality')}) ===")
        m_ok = download_to(onnx_url, onnx_dst, min_bytes=onnx_min)
        c_ok = download_to(json_url, json_dst, min_bytes=json_min)

        if m_ok and c_ok:
            ok += 1
        else:
            # cleanup partial
            if onnx_dst.exists(): onnx_dst.unlink()
            if json_dst.exists(): json_dst.unlink()

    print(f"\nDone. Downloaded {ok} voice(s) into {VOICES_DIR}")
    if ok == 0:
        raise SystemExit("No voices downloaded. Check connectivity or repo paths.")

if __name__ == "__main__":
    main()

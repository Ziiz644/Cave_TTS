import os
import json
import random
import urllib.request
import urllib.error
from pathlib import Path

VOICES_DIR = Path(os.environ.get("VOICES_DIR", "/app/voices"))
VOICES_DIR.mkdir(parents=True, exist_ok=True)

# Piper voices index (authoritative)
VOICES_INDEX_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/voices.json?download=true"

# Base URL for actual files in the same release tag
BASE_RESOLVE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"

VOICE_COUNT = int(os.environ.get("VOICE_COUNT", "30"))
# We will pick English + Arabic Jordan (ar_JO) as “Arabic bucket”
# You can later extend to other Arabic locales if/when they exist.
LANG_ALLOW_PREFIXES = os.environ.get("VOICE_LANG_PREFIXES", "en/,ar/").split(",")

def http_get_bytes(url: str, timeout=180) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()

def download_file(url: str, dst: Path, min_bytes: int) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = http_get_bytes(url)
        if len(data) < min_bytes:
            print(f"❌ Too small ({len(data)} bytes) for {url}")
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
    index_bytes = http_get_bytes(VOICES_INDEX_URL)
    voices_index = json.loads(index_bytes.decode("utf-8"))

    # voices.json keys include both .onnx and .onnx.json entries, e.g.
    # "en/en_GB/alan/low/en_GB-alan-low.onnx"
    # "en/en_GB/alan/low/en_GB-alan-low.onnx.json"
    onnx_keys = [k for k in voices_index.keys() if k.endswith(".onnx")]

    # Filter by desired language buckets (en/ and ar/)
    def allowed(k: str) -> bool:
        return any(k.startswith(prefix) for prefix in LANG_ALLOW_PREFIXES)

    candidates = [k for k in onnx_keys if allowed(k)]
    if not candidates:
        raise SystemExit("No voice candidates after filtering. Check VOICE_LANG_PREFIXES.")

    random.shuffle(candidates)
    selected = candidates[:VOICE_COUNT]

    ok_models = 0
    for onnx_key in selected:
        json_key = onnx_key + ".json"
        if json_key not in voices_index:
            print(f"❌ Missing config entry in index for {onnx_key}")
            continue

        model_url = BASE_RESOLVE + onnx_key + "?download=true"
        cfg_url   = BASE_RESOLVE + json_key + "?download=true"

        # Save files exactly as their path suggests
        model_dst = VOICES_DIR / onnx_key
        cfg_dst   = VOICES_DIR / json_key

        print(f"\n=== {onnx_key} ===")

        # .onnx is large (~MBs), .json is small (~KBs). Use different thresholds.
        m_ok = download_file(model_url, model_dst, min_bytes=500_000)   # >= ~500KB
        c_ok = download_file(cfg_url,   cfg_dst,   min_bytes=500)       # >= 500 bytes

        if m_ok and c_ok:
            ok_models += 1
        else:
            # cleanup partials
            if model_dst.exists(): model_dst.unlink()
            if cfg_dst.exists(): cfg_dst.unlink()

    print(f"\nDone. Downloaded {ok_models} voice(s) into {VOICES_DIR}")

    if ok_models == 0:
        raise SystemExit("No voices downloaded. Check network / index / filters.")

if __name__ == "__main__":
    main()

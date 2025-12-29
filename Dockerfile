FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VOICES_DIR=/app/voices

# Minimal system deps:
# - espeak-ng commonly needed by local TTS stacks (phonemization)
# - ca-certificates/curl for downloading voices during build
# - libsndfile1 is harmless and useful if you add audio tooling later
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    ca-certificates \
    curl \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app
COPY . /app

# (Optional but recommended) download voices during build
# If you use this, keep your selected voices list small for Render build speed.
RUN python /app/scripts/download_voices.py

EXPOSE 8000
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VOICES_DIR=/app/voices

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl unzip \
    espeak-ng \
 && rm -rf /var/lib/apt/lists/*

# ---- install Piper CLI (binary) ----
# Choose a stable Piper release build for Linux x86_64
# (This puts `piper` on PATH)
RUN mkdir -p /opt/piper && \
    curl -L -o /tmp/piper.zip https://github.com/rhasspy/piper/releases/latest/download/piper_linux_x86_64.zip && \
    unzip /tmp/piper.zip -d /opt/piper && \
    ln -s /opt/piper/piper /usr/local/bin/piper && \
    rm -f /tmp/piper.zip

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# If you have download_voices.py enabled:
RUN python /app/scripts/download_voices.py

EXPOSE 8000
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VOICES_DIR=/app/voices

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl unzip \
    espeak-ng \
 && rm -rf /var/lib/apt/lists/*

# ---- install Piper CLI (binary) ----
RUN mkdir -p /opt/piper && \
    curl -fL -o /tmp/piper.tar.gz https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz && \
    tar -xzf /tmp/piper.tar.gz -C /opt/piper --strip-components=1 && \
    ln -sf /opt/piper/piper /usr/local/bin/piper && \
    rm -f /tmp/piper.tar.gz && \
    piper --help | head -n 2


WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# If you have download_voices.py enabled:
RUN python /app/scripts/download_voices.py

EXPOSE 8000
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]

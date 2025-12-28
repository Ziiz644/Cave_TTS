FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# -------------------------
# 1️⃣ System + build deps
# -------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -------------------------
# 2️⃣ Python deps
# -------------------------
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY server.py /app/server.py

EXPOSE 8008

CMD ["uvicorn", "server:app","--host","0.0.0.0","--port","8008","]

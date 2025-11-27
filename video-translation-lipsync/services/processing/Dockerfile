# services/processing/Dockerfile
FROM python:3.10-slim

# Build args: set INSTALL_WAV2LIP=true to clone Wav2Lip and attempt to download checkpoint.
ARG INSTALL_WAV2LIP=false
ARG WAV2LIP_CHECKPOINT_URL=""

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY pipeline.py .

RUN mkdir -p /app/uploads /app/outputs /app/models

# Optionally clone Wav2Lip and download the checkpoint at build time when enabled.
RUN if [ "$INSTALL_WAV2LIP" = "true" ] ; then \
            echo "INSTALL_WAV2LIP=true: cloning Wav2Lip..."; \
            cd /app/models && git clone https://github.com/Rudrabha/Wav2Lip.git Wav2Lip || true; \
            mkdir -p /app/models/Wav2Lip/checkpoints; \
            if [ -n "$WAV2LIP_CHECKPOINT_URL" ]; then \
                echo "Downloading Wav2Lip checkpoint..."; \
                apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*; \
                curl -L -o /app/models/Wav2Lip/checkpoints/wav2lip_gan.pth "$WAV2LIP_CHECKPOINT_URL" || echo "checkpoint download failed"; \
            else \
                echo "No WAV2LIP_CHECKPOINT_URL provided; skipping checkpoint download."; \
            fi; \
        else echo "INSTALL_WAV2LIP not enabled; skipping Wav2Lip clone."; fi

EXPOSE 8002

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
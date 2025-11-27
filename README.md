# Real-Time Voice Translation + Lip-Synced Video

A functional prototype that performs voice translation of uploaded videos and produces lip-synced output using locally stored pretrained models.

## Overview

This system implements a complete pipeline for:
1. Extracting transcript and timestamps from input audio (ASR)
2. Translating text to target language
3. Converting translated text to target-language speech (TTS)
4. Syncing lips in the uploaded video with generated audio (Lip-Sync)

## Architecture

Three FastAPI microservices with Docker and Kubernetes support:
- **Upload Service** (Port 8001) - Handles video/audio uploads with chunked upload support
- **Processing Service** (Port 8002) - Runs the AI pipeline
- **Media Service** (Port 8003) - Streams/downloads generated videos

## Setup and Run Instructions

### Prerequisites
- Docker & Docker Compose
- 8GB+ RAM recommended
- 10GB+ disk space for models

### Quick Start with Docker Compose

```bash
# 1. Clone and navigate to project
git clone <repository-url>
cd video-translation-lipsync

# 2. Start all services
docker-compose up --build -d

# 3. Verify services are running
curl http://localhost:8001/
curl http://localhost:8002/
curl http://localhost:8003/
```

### Running with Kubernetes

```bash
# 1. Build Docker images
docker build -t video-pipeline/upload:latest ./services/upload
docker build -t video-pipeline/processing:latest ./services/processing
docker build -t video-pipeline/media:latest ./services/media

# 2. Deploy to Kubernetes
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/upload-deployment.yaml
kubectl apply -f k8s/processing-deployment.yaml
kubectl apply -f k8s/media-deployment.yaml
kubectl apply -f k8s/services.yaml

# 3. Port forward to access services
kubectl port-forward -n video-pipeline svc/upload-service 8001:8001
kubectl port-forward -n video-pipeline svc/processing-service 8002:8002
kubectl port-forward -n video-pipeline svc/media-service 8003:8003
```

## API Examples

### 1. Upload Endpoint

**Upload a video file:**
```bash
curl -X POST "http://localhost:8001/upload" \
  -F "file=@input_video.mp4" \
  -F "target_language=es"
```

**Response:**
```json
{
  "upload_id": "abc123-def456",
  "filename": "input_video.mp4",
  "size": 52428800,
  "target_language": "es",
  "status": "uploaded"
}
```

**Check upload status:**
```bash
curl "http://localhost:8001/upload/abc123-def456/status"
```

### 2. Process Endpoint

**Start processing:**
```bash
curl -X POST "http://localhost:8002/process" \
  -H "Content-Type: application/json" \
  -d '{
    "upload_id": "abc123-def456",
    "target_language": "es"
  }'
```

**Response:**
```json
{
  "job_id": "job_xyz789",
  "status": "queued",
  "message": "Processing started"
}
```

**Check processing status:**
```bash
curl "http://localhost:8002/process/job_xyz789/status"
```

**Response:**
```json
{
  "job_id": "job_xyz789",
  "status": "processing",
  "progress": 60,
  "created_at": "2024-01-15 10:30:00"
}
```

### 3. Media Endpoint

**Download generated video:**
```bash
curl "http://localhost:8003/media/job_xyz789" -o output.mp4
```

**Stream video:**
```bash
curl "http://localhost:8003/media/job_xyz789?stream=true"
```

**Get media info:**
```bash
curl "http://localhost:8003/media/job_xyz789/info"
```

## Pretrained Models Used

### ASR (Automatic Speech Recognition)
- **Model**: OpenAI Whisper (base)
- **Purpose**: Extract transcript and timestamps from audio
- **Location**: Auto-downloaded to `/app/models/`

### Translation
- **Model**: MarianMT (Helsinki-NLP/opus-mt-en-{target})
- **Purpose**: Translate text to target language
- **Location**: Auto-downloaded on first use
- **Supported Languages**: es (Spanish), fr (French), de (German), it (Italian), pt (Portuguese)

### TTS (Text-to-Speech)
- **Model**: Coqui TTS (Tacotron2-DDC)
- **Purpose**: Convert translated text to speech
- **Location**: Auto-downloaded to model cache

### Lip-Sync
- **Model**: Wav2Lip
- **Purpose**: Sync lips with generated audio
- **Location**: `/app/models/Wav2Lip/` (optional - falls back to audio replacement)
- **Note**: For full lip-sync, clone Wav2Lip repository into models directory

## Where to Place Models

Models are automatically downloaded on first run. No manual placement required.

**For manual model setup (optional):**

```bash
# Models will be cached in:
./models/whisper/          # Whisper ASR models
./models/transformers/     # MarianMT translation models
./models/tts/              # Coqui TTS models
./models/Wav2Lip/          # Wav2Lip (if manually installed)
```

**To pre-download models:**
```bash
docker-compose run processing python pipeline.py
```

## Project Structure

```
video-translation-lipsync/
├── README.md
├── docker-compose.yml
│
├── services/
│   ├── upload/
│   │   ├── app.py                 # Upload service implementation
│   │   ├── requirements.txt       # Python dependencies
│   │   └── Dockerfile
│   │
│   ├── processing/
│   │   ├── app.py                 # Processing service API
│   │   ├── pipeline.py            # AI pipeline implementation
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   │
│   └── media/
│       ├── app.py                 # Media streaming service
│       ├── requirements.txt
│       └── Dockerfile
│
├── k8s/
│   ├── namespace.yaml             # Kubernetes namespace
│   ├── pvc.yaml                   # Persistent volume claims
│   ├── upload-deployment.yaml     # Upload service deployment
│   ├── processing-deployment.yaml # Processing service deployment
│   ├── media-deployment.yaml      # Media service deployment
│   └── services.yaml              # Kubernetes services
│
├── uploads/                       # Uploaded files storage
├── outputs/                       # Processed videos storage
└── models/                        # Model cache directory
```

## Configuration

### Environment Variables

**Upload Service:**
- `UPLOAD_DIR`: Upload directory (default: `/app/uploads`)
- `MAX_FILE_SIZE`: Maximum file size in bytes (default: `104857600` = 100MB)
- `CHUNK_SIZE`: Upload chunk size (default: `8388608` = 8MB)

**Processing Service:**
- `OUTPUT_DIR`: Output directory (default: `/app/outputs`)
- `UPLOAD_DIR`: Upload directory (default: `/app/uploads`)
- `MODEL_DIR`: Model cache directory (default: `/app/models`)
- `WHISPER_MODEL`: Whisper model size (default: `base`, options: `tiny`, `base`, `small`, `medium`, `large`)
- `USE_GPU`: Enable GPU acceleration (default: `false`)

## Notes & Important Implementation Details

- Translation model coverage: The pipeline will attempt to load a Helsinki-NLP Marian model for `en -> {target}` when available (for many common languages). If a language-specific Helsinki model isn't available, the pipeline falls back to a multilingual seq2seq model (`facebook/m2m100_418M`) which supports many languages — however this model is larger and may need more memory.

- Tokenization and chunking: Text is chunked by tokenizer tokens (max 512 tokens) rather than naive sentence splits to avoid silent truncation that would change translation output.

- Audio length alignment: The generated TTS audio is adjusted (trimmed/padded or resampled) to approximately match the original audio duration before lip-sync. This improves lip-sync quality, but perfect timing may still require manual tuning for extreme duration mismatches.

- Speaker cloning: You can provide a speaker reference audio by setting `MODEL_SPEAKER_REF` environment variable to a local WAV file path. The pipeline will try to use it with Coqui TTS (when supported by the chosen TTS model) to approximate the original speaker's voice. This is optional and may not be perfect for all voices.

- Wav2Lip installation: Wav2Lip is not auto-installed in the container due to its GitHub repo size and checkpoint licensing. To enable full lip-sync:
  1. Clone Wav2Lip into the models folder: `git clone https://github.com/Rudrabha/Wav2Lip.git models/Wav2Lip`
  2. Download the checkpoint and place it at: `models/Wav2Lip/checkpoints/wav2lip_gan.pth`
  3. Ensure the `models/Wav2Lip/inference.py` file exists (the pipeline uses it when present).

If Wav2Lip isn't available the pipeline will fallback to combining the generated audio with the original video (no lip movement correction).

### Optional: Build processing image with Wav2Lip included

If you want the Docker image to clone the Wav2Lip repository and attempt to download a checkpoint at build time, you can pass build-time arguments. Be aware this will increase image size and may take a long time.

Example (PowerShell):

```powershell
# Replace <checkpoint_url> with a hosted URL to the wav2lip_gan.pth file if you have one.
docker build `
  --build-arg INSTALL_WAV2LIP=true `
  --build-arg WAV2LIP_CHECKPOINT_URL="<checkpoint_url>" `
  -t video-pipeline/processing:with-wav2lip `
  -f services/processing/Dockerfile .
```

If you do not have a public checkpoint URL, you can clone Wav2Lip into your local `models/` directory and mount it as a volume into the container at runtime instead. This avoids embedding the checkpoint into the image.

Warning:
- The Wav2Lip repository and checkpoint are large. Building with `INSTALL_WAV2LIP=true` will significantly increase image size.
- Check checkpoint licensing and distribution restrictions before including it in images or a public registry.

**Media Service:**
- `MEDIA_DIR`: Media files directory (default: `/app/outputs`)

### GPU Support

To enable GPU acceleration:

**Edit docker-compose.yml:**
```yaml
processing:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  environment:
    - USE_GPU=true
```

**Requires:** NVIDIA Docker runtime

## Supported File Formats

**Input:**
- Video: `.mp4`, `.avi`, `.mov`, `.mkv`
- Audio: `.mp3`, `.wav`, `.m4a`

**Output:**
- Video: `.mp4` (H.264)

## Pipeline Stages and Progress

The processing pipeline has 5 stages with progress tracking:

1. **Audio Extraction** (0-20%): Extract audio from video
2. **Transcription** (20-40%): ASR with Whisper
3. **Translation** (40-60%): Translate text with MarianMT
4. **Speech Generation** (60-80%): TTS with Coqui
5. **Lip Synchronization** (80-100%): Sync with Wav2Lip

## Testing

### Automated Test Script

```bash
# Make script executable
chmod +x test_pipeline.sh

# Run test (requires test_video.mp4 in root directory)
./test_pipeline.sh
```

### Manual Testing

```bash
# 1. Upload
UPLOAD_ID=$(curl -s -X POST "http://localhost:8001/upload" \
  -F "file=@test_video.mp4" \
  -F "target_language=es" | jq -r '.upload_id')

# 2. Process
JOB_ID=$(curl -s -X POST "http://localhost:8002/process" \
  -H "Content-Type: application/json" \
  -d "{\"upload_id\": \"$UPLOAD_ID\", \"target_language\": \"es\"}" \
  | jq -r '.job_id')

# 3. Monitor
while true; do
  STATUS=$(curl -s "http://localhost:8002/process/$JOB_ID/status" | jq -r '.status')
  echo "Status: $STATUS"
  [ "$STATUS" = "completed" ] && break
  sleep 5
done

# 4. Download
curl "http://localhost:8003/media/$JOB_ID" -o output.mp4
```

## Troubleshooting

**Services won't start:**
```bash
# Check logs
docker-compose logs -f

# Check individual service
docker-compose logs -f processing
```

**Out of memory:**
- Increase Docker memory limit to 8GB+
- Use smaller Whisper model: `WHISPER_MODEL=tiny`

**Models not downloading:**
```bash
# Manually trigger download
docker-compose run processing python pipeline.py
```

**Slow processing:**
- Enable GPU support (3-5x faster)
- Use smaller models
- Process shorter videos first

## Performance

**Typical Processing Times (CPU):**
- 30 second video: ~2-3 minutes
- 1 minute video: ~5-7 minutes
- 2 minute video: ~10-15 minutes

**With GPU:**
- 3-5x faster than CPU

## Notes

- First run will download models (~2-5GB), taking 5-10 minutes
- Supports video/audio uploads up to 100MB+ with chunked upload
- All processing is async with job tracking
- Models run locally, no cloud APIs used
- Optional GPU support for faster processing

## Docker Images

The project creates three Docker images:
- `video-pipeline/upload:latest` - Upload service
- `video-pipeline/processing:latest` - Processing service  
- `video-pipeline/media:latest` - Media service

## Kubernetes Resources

- **Namespace**: `video-pipeline`
- **PVCs**: `uploads-pvc` (10Gi), `outputs-pvc` (20Gi), `models-pvc` (5Gi)
- **Services**: ClusterIP for inter-service communication
- **Deployments**: 2 replicas for upload/media, 1 for processing

## License

MIT

# Rakshak AI Surveillance Platform

Rakshak is a full-stack smart surveillance system for real-time threat monitoring. It combines a FastAPI backend, OpenCV video ingestion, YOLO-based detection models, SQLite alert history, optional Ollama scene enrichment, optional SMTP/SMS notification routing, and a React + TypeScript dashboard for live camera telemetry.

The platform can detect and track:

- Weapons
- Fire and smoke
- Falls or possible injuries
- Crowd density
- Restricted-zone intrusion
- Suspicious movement around configured zones

## Features

- Live camera or video-file monitoring over WebSocket telemetry.
- Multi-model inference using bundled YOLO weights in `backend/models/`.
- Persistent alert history with snapshot evidence.
- Configurable detection thresholds, active modules, source settings, and notification routing from the dashboard.
- Restricted-zone polygon tools with optional authorized-person bypass rules.
- ByteTrack-based person tracking, temporal validation, scene memory, threat scoring, and controlled multi-agent analysis.
- Docker Compose deployment with Redis-backed scene memory.

## Tech Stack

- Backend: Python, FastAPI, Uvicorn, OpenCV, Ultralytics YOLO, SQLite, Redis.
- Frontend: React, TypeScript, Vite, Tailwind CSS, lucide-react.
- Optional integrations: Ollama for local vision-language alert descriptions, SMTP for email, Twilio for SMS.

## Repository Structure

```text
.
├── backend/
│   ├── ai_core/              # Detectors, tracking, validation, threat scoring, memory, agents
│   ├── ai_modules/           # Compatibility facade and perimeter engine
│   ├── eval/                 # Detector evaluation utility
│   ├── models/               # YOLO model weights used by the app
│   ├── services/             # Notification and enrichment services
│   ├── config.py
│   ├── database.py
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── TECHNICAL_BLUEPRINT.md
└── EXECUTIVE_PRESENTATION.md
```

Runtime files such as `.env`, `alerts.db`, `snapshots/`, `uploads/`, virtual environments, `node_modules/`, and frontend build output are intentionally ignored.

## Prerequisites

- Python 3.10 or newer
- Node.js 20 or newer
- npm
- Docker Desktop or Docker Engine, only if using Docker
- A webcam, video file, or accessible camera source
- Optional: Ollama for local AI descriptions

## Local Installation

Clone the project and enter the repository:

```bash
git clone <your-repository-url>
cd Rakshak
```

Create a local environment file:

```bash
cp .env.example .env
```

On Windows PowerShell, use:

```powershell
Copy-Item .env.example .env
```

### Backend Setup

Create and activate a Python virtual environment:

```bash
python -m venv backend/.venv
source backend/.venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
```

Install backend dependencies:

```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

Start the API server:

```bash
python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Useful backend URLs:

- API health check: `http://localhost:8000/api/health`
- API docs: `http://localhost:8000/docs`
- WebSocket telemetry: `ws://localhost:8000/api/ws/telemetry`

### Frontend Setup

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite development URL, usually `http://localhost:5173`.

The frontend automatically talks to `http://localhost:8000` when it is running on Vite port `5173`. To override the backend URL:

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

On Windows PowerShell:

```powershell
$env:VITE_API_BASE_URL="http://localhost:8000"
npm run dev
```

## Optional Ollama Enrichment

Rakshak works without Ollama. When Ollama is running, alert descriptions can be enriched with local vision-language context.

```bash
ollama pull moondream
ollama serve
```

For local non-Docker runs, set:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=moondream
```

For Docker Desktop, `.env.example` uses `http://host.docker.internal:11434`.

## Configuration

Most runtime settings can be changed in `.env` or through the dashboard.

Important environment variables:

- `SOURCE_TYPE`: `webcam` or `file`.
- `CAMERA_DEVICE_ID`: Webcam index, usually `0`.
- `VIDEO_FILEPATH`: Video path when `SOURCE_TYPE=file`.
- `FRAME_WIDTH`, `FRAME_HEIGHT`: Capture resize dimensions.
- `TARGET_FPS`, `INFERENCE_FPS`: Streaming and inference cadence.
- `MODEL_IMGSZ`, `DETECTION_IOU`: YOLO inference settings.
- `WEAPON_MODEL_FILE`, `FIRE_MODEL_FILE`, `FALL_MODEL_FILE`, `PERSON_MODEL_FILE`: Model files inside `backend/models/`.
- `ACTIVE_MODULES`: Comma-separated enabled modules.
- `CONFIDENCE_*`, `ALERT_SCORE_*`, `INSTANT_CONFIDENCE_*`: Detection and alert thresholds.
- `TRACKING_ENABLED`, `TRACKER_CONFIG`: Person tracking settings.
- `SCENE_MEMORY_BACKEND`: `memory` for local runs or `redis` for shared state.
- `MULTI_AGENT_ENABLED`, `MULTI_AGENT_FRAMEWORK`: Controlled analysis workflow settings.
- `AUTHORIZED_PEOPLE`: JSON roster for restricted-zone bypass.
- `SMTP_ENABLED`, `SMTP_SERVER`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `TO_ADDRESS`: Email alerts.
- `SMS_ENABLED`, `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, `TO_PHONE`: SMS alerts.
- `CORS_ORIGINS`: Comma-separated allowed frontend origins.

Keep real credentials only in `.env`. Do not commit them.

## Dashboard Workflow

1. Open the dashboard and view the primary stream.
2. Toggle detection modules for weapon, fire, fall, crowd, and intrusion monitoring.
3. Draw or update intrusion zones on the camera viewport.
4. Add authorized people who can bypass restricted-zone alerts when enabled and present.
5. Tune confidence thresholds and alert scoring from the analytics controls.
6. Configure notification routing for email or SMS.
7. Review recent anomalies and open snapshot evidence from the alert history.

## Run With Docker

The production Docker image serves the FastAPI backend and compiled React frontend from one container. Redis is included through Docker Compose for scene memory.

```bash
cp .env.example .env
docker compose up --build -d
```

Open:

- Dashboard: `http://localhost:8000`
- Health check: `http://localhost:8000/api/health`

Useful commands:

```bash
docker compose ps
docker compose logs -f rakshak-platform
docker compose down
```

Runtime data is stored in the Docker volume `rakshak-data`:

- `/data/alerts.db`
- `/data/snapshots`
- `/data/uploads`

For Docker Desktop on Windows or macOS, webcam passthrough is usually unavailable. Use a video file source for demos:

```env
SOURCE_TYPE=file
VIDEO_FILEPATH=/data/uploads/demo.mp4
```

On Linux hosts with a physical camera, add a device mapping to the `rakshak-platform` service:

```yaml
devices:
  - /dev/video0:/dev/video0
```

## Detection Evaluation

Use `backend/eval/evaluate_detection.py` with a YOLO-format dataset:

```text
dataset/
  images/
    frame_001.jpg
  labels/
    frame_001.txt
```

Default class mapping:

- `0`: `weapon_detection`
- `1`: `fire_detection`
- `2`: `fall_detection`
- `3`: `person`

Run evaluation:

```bash
python backend/eval/evaluate_detection.py --dataset /path/to/dataset --output backend/eval/runs/baseline
```

Outputs include `metrics.json`, `metrics.csv`, and false-positive / false-negative crops.

## Verification

Backend syntax check:

```bash
python -m py_compile backend/main.py backend/database.py backend/config.py backend/ai_modules/detection_engine.py backend/ai_modules/perimeter_engine.py backend/services/notifier.py
```

Frontend checks:

```bash
cd frontend
npm run lint
npm run build
```

## GitHub Upload Notes

This repository is prepared so generated artifacts and private runtime data stay out of Git. Before pushing, confirm the staged files look right:

```bash
git status --short
```

The bundled model weights are below GitHub's 100 MB per-file limit. If you later replace them with larger files, use Git LFS or host the weights separately and document the download location.

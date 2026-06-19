# --- STAGE 1: Build Vite React Frontend ---
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

# Copy frontend configuration files
COPY frontend/package*.json ./
COPY frontend/tsconfig*.json ./
COPY frontend/vite.config.ts ./
COPY frontend/tailwind.config.js ./
COPY frontend/postcss.config.js ./
COPY frontend/index.html ./

# Install dependencies
RUN npm ci

# Copy frontend source files
COPY frontend/src ./src
COPY frontend/public ./public

# Build production distribution bundle
RUN npm run build

# --- STAGE 2: Build FastAPI + Python Inference Backend ---
FROM python:3.10-slim AS backend-runner
WORKDIR /app

# Install system dependencies (needed for OpenCV image manipulation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install dependencies
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend application source
COPY backend /app/backend

# Copy production frontend build from Stage 1
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Expose server ingress port
EXPOSE 8000

# Set environment paths
ENV FRONTEND_DIST_DIR=/app/frontend/dist
ENV PROJECT_ROOT=/app
ENV ALERTS_DB_PATH=/data/alerts.db
ENV SNAPSHOTS_DIR=/data/snapshots
ENV UPLOADS_DIR=/data/uploads
ENV SOURCE_TYPE=webcam
ENV CAMERA_DEVICE_ID=0

WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

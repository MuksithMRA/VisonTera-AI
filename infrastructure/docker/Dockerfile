# ──────────────────────────────────────────────────────────────────────────────
#  VisionTera AI — Dockerfile
#  Multi-camera YOLO person detection + gender classification + Re-ID
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.9-slim

WORKDIR /app

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV runtime
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Video / codec support
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    # Image format support
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    # GStreamer (RTSP streams)
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    # GTK (optional, for headless OpenCV)
    libgtk-3-0 \
    # Math / compiler
    gfortran \
    # Network tools
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application source ───────────────────────────────────────────────────────
# Copy the full application source tree
COPY app/            ./app/
COPY static/         ./static/
COPY web_app.py      ./web_app.py
COPY index.html      ./index.html
COPY login.html      ./login.html

# ── Model weights ────────────────────────────────────────────────────────────
# Base YOLO detection models
COPY infrastructure/models/yolo11m.pt          ./infrastructure/models/yolo11m.pt
COPY infrastructure/models/yolo11n.pt          ./infrastructure/models/yolo11n.pt

# Gender classification models
COPY infrastructure/models/resnet50-gender.pt  ./infrastructure/models/resnet50-gender.pt
COPY infrastructure/models/yolo11m-gender.pt   ./infrastructure/models/yolo11m-gender.pt

# Latest crowd-optimised detection model (crowd_v_*)
COPY infrastructure/models/crowd_v_20260219_102246/ ./infrastructure/models/crowd_v_20260219_102246/

# Latest gender versioned model (v_*) – pick latest trained
COPY infrastructure/models/v_20260224_075137/  ./infrastructure/models/v_20260224_075137/

# Re-ID feature extractor weights directory (optional — uses ImageNet fallback if empty)
# Creates the directory; place reid weights here before building to enable Re-ID
RUN mkdir -p ./infrastructure/models/reid

# ── Runtime user (non-root for security) ────────────────────────────────────
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# ── Expose FastAPI port ──────────────────────────────────────────────────────
EXPOSE 8000

# ── Healthcheck ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/status || exit 1

# ── Entrypoint ───────────────────────────────────────────────────────────────
# Use uvicorn directly (no reload in production)
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
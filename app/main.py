from app.utils import patch_torch_load
patch_torch_load()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from app.config import AppConfig, logger
from app.services.state import engine
from app.controllers.base_controller import router as base_router
from app.controllers.detection_controller import router as detection_router
from app.controllers.training_controller import router as training_router

tags_metadata = [
    {
        "name": "Dashboard",
        "description": "Dashboard and status endpoints",
    },
    {
        "name": "Detection",
        "description": "Real-time person detection operations including start/stop detection and video streaming",
    },
    {
        "name": "Training",
        "description": "Model training and retraining operations",
    },
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load_model()
    logger.info("Application started")
    yield
    logger.info("Application shutdown")

app = FastAPI(
    lifespan=lifespan,
    title="VisionTera API",
    description="""
## VisionTera - YOLO Person Detection API

This API provides real-time person detection capabilities using YOLO models.

### Features:
* **Real-time Detection**: Start/stop person detection from camera feeds
* **Video Streaming**: MJPEG video stream with detection overlays
* **WebSocket Stats**: Real-time statistics via WebSocket connection
* **Model Training**: Retrain detection models with new data

### WebSocket Endpoints:
* **`/ws/stats`**: Real-time detection statistics (FPS, person count, coordinates)
""",
    version="1.0.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "VisionTera Support",
    },
    license_info={
        "name": "MIT",
    },
)

app.mount("/static", StaticFiles(directory=AppConfig.BASE_DIR / "static"), name="static")

app.include_router(base_router)
app.include_router(detection_router)
app.include_router(training_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from app.utils import patch_torch_load
patch_torch_load()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from app.config import AppConfig, logger
from app.services.state import camera_manager
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
        "description": "Multi-camera real-time person detection with parallel processing",
    },
    {
        "name": "Training",
        "description": "Model training and retraining operations",
    },
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    camera_manager.load_models()
    logger.info("Application started - Multi-camera detection ready")
    yield
    await camera_manager.stop_all_cameras()
    logger.info("Application shutdown - All cameras stopped")

app = FastAPI(
    lifespan=lifespan,
    title="VisionTera API",
    description="""
## VisionTera - Multi-Camera YOLO Person Detection API

This API provides real-time person detection capabilities with **parallel multi-camera processing**.

### Key Features:
* **Multi-Camera Support**: Run detection on multiple cameras simultaneously
* **Parallel Processing**: Each camera runs in its own thread with shared GPU inference
* **Real-time Detection**: YOLO-based person detection with gender classification
* **Auto-Reconnection**: Automatic camera reconnection on failures
* **Per-Camera Stats**: Individual statistics for each camera

### Multi-Camera Endpoints:
* **POST `/api/camera/start`**: Start a specific camera with unique ID
* **POST `/api/camera/{camera_id}/stop`**: Stop a specific camera
* **GET `/api/cameras/active`**: Get all active cameras and their statuses
* **GET `/video_feed/{camera_id}`**: Video stream for a specific camera

### WebSocket Endpoints:
* **`/ws/stats/{camera_id}`**: Real-time stats for a specific camera
* **`/ws/stats`**: Aggregated stats for all cameras
""",
    version="2.0.0",
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

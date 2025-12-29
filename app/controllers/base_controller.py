from fastapi import APIRouter
from fastapi.responses import FileResponse
from app.config import AppConfig
from app.services.state import engine
from app.models.schemas import DetectionStatus

router = APIRouter(tags=["Dashboard"])

@router.get(
    "/",
    summary="Get Dashboard",
    description="Returns the main dashboard HTML page for the VisionTera application."
)
async def get_dashboard():
    return FileResponse(AppConfig.BASE_DIR / "index.html", media_type="text/html")

@router.get(
    "/status",
    response_model=DetectionStatus,
    summary="Get System Status",
    description="Returns the current status of the detection system including whether detection is running, frames processed, and the active camera index."
)
async def get_status():
    return {
        "status": "ok", 
        "running": engine.is_running,
        "frames_processed": engine.frame_count,
        "camera_index": engine.camera_index
    }

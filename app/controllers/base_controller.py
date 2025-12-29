from fastapi import APIRouter
from fastapi.responses import FileResponse
from app.config import AppConfig
from app.services.state import engine
from app.models.schemas import DetectionStatus

router = APIRouter()

@router.get("/")
async def get_dashboard():
    return FileResponse(AppConfig.BASE_DIR / "index.html", media_type="text/html")

@router.get("/status", response_model=DetectionStatus)
async def get_status():
    return {
        "status": "ok", 
        "running": engine.is_running,
        "frames_processed": engine.frame_count,
        "camera_index": engine.camera_index
    }

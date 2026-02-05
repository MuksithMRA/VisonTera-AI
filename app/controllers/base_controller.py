from fastapi import APIRouter
from fastapi.responses import FileResponse
from app.config import AppConfig
from app.services.state import camera_manager
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
    description="Returns the current status of the detection system including active cameras and running state."
)
async def get_status():
    return DetectionStatus(
        status="ok",
        running=camera_manager.active_camera_count > 0,
        active_cameras=camera_manager.active_camera_count,
        total_cameras=camera_manager.total_camera_count
    )

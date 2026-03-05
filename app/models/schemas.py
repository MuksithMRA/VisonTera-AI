from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class BoundingBoxSchema(BaseModel):
    x1: float = Field(..., description="Left x coordinate")
    y1: float = Field(..., description="Top y coordinate")
    x2: float = Field(..., description="Right x coordinate")
    y2: float = Field(..., description="Bottom y coordinate")


class DetectionSchema(BaseModel):
    id: int = Field(..., description="Track ID of the detection (per-camera)")
    global_id: int = Field(-1, description="Cross-camera unique person ID from Re-ID")
    x: float = Field(..., description="Center x coordinate")
    y: float = Field(..., description="Bottom y coordinate")
    confidence: float = Field(..., description="Detection confidence")
    gender: str = Field(..., description="Gender classification: Male, Female, or Person")
    bbox: BoundingBoxSchema = Field(..., description="Bounding box coordinates")


class CameraStatusSchema(BaseModel):
    camera_id: str = Field(..., description="Unique camera identifier")
    state: str = Field(..., description="Current camera state")
    name: str = Field(..., description="Camera display name")
    source: str = Field(..., description="Camera source (index or URL)")
    camera_type: str = Field(..., description="Camera type: local or rtsp")
    frames_processed: int = Field(..., description="Total frames processed")
    error_message: Optional[str] = Field(None, description="Error message if any")


class CameraStartRequest(BaseModel):
    camera_id: str = Field(..., description="Unique identifier for this camera session")
    source: str = Field(..., description="Camera index or RTSP URL")
    name: Optional[str] = Field(None, description="Display name for the camera")
    backend_camera_id: Optional[int] = Field(None, description="Backend ID to map to this camera")
    confidence: float = Field(0.5, ge=0.1, le=1.0, description="Detection confidence threshold")
    show_coords: bool = Field(True, description="Show coordinates on detections")
    show_fps: bool = Field(True, description="Show FPS counter")
    box_color: str = Field("00FF88", description="Box color in hex format")
    counting_line: Optional[List[List[float]]] = Field(None, description="List of [x, y] coordinates for counting boundary [[x1, y1], [x2, y2]]")

    model_config = {
        "json_schema_extra": {
            "example": {
                "camera_id": "cam_1",
                "source": "0",
                "name": "Main Entrance",
                "backend_camera_id": 12,
                "confidence": 0.5,
                "show_coords": True,
                "show_fps": True,
                "box_color": "00FF88"
            }
        }
    }


class CameraResponse(BaseModel):
    status: str = Field(..., description="Response status")
    camera_id: str = Field(..., description="Camera identifier")
    message: Optional[str] = Field(None, description="Additional message")


class MultiCameraStatusResponse(BaseModel):
    total_cameras: int = Field(..., description="Total number of active cameras")
    cameras: List[Dict[str, Any]] = Field(..., description="List of camera statuses")


class ProcessingStatsSchema(BaseModel):
    camera_id: str = Field(..., description="Camera identifier")
    fps: float = Field(..., description="Current frames per second")
    width: int = Field(..., description="Frame width")
    height: int = Field(..., description="Frame height")
    person_count: int = Field(..., description="Number of persons detected")
    male_count: int = Field(..., description="Number of males detected")
    female_count: int = Field(..., description="Number of females detected")
    detections: List[Dict[str, Any]] = Field(..., description="List of detections")
    timestamp: str = Field(..., description="ISO timestamp")


class AllStatsResponse(BaseModel):
    type: str = Field("all_stats", description="Message type")
    cameras: List[ProcessingStatsSchema] = Field(..., description="Stats for all cameras")


class DetectionStatus(BaseModel):
    status: str = Field(..., description="System status indicator", example="ok")
    running: bool = Field(..., description="Whether detection is currently running", example=True)
    active_cameras: int = Field(..., description="Number of active cameras")
    total_cameras: int = Field(..., description="Total cameras being managed")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "running": True,
                "active_cameras": 2,
                "total_cameras": 2
            }
        }
    }


class StartResponse(BaseModel):
    status: str = Field(..., description="Response status: 'started', 'already_running', or 'error'")
    camera_id: Optional[str] = Field(None, description="Camera identifier")
    message: Optional[str] = Field(None, description="Additional message")


class StopResponse(BaseModel):
    status: str = Field(..., description="Response status")
    camera_id: Optional[str] = Field(None, description="Camera identifier")
    count: Optional[int] = Field(None, description="Number of cameras stopped (for stop_all)")


class TrainingStatus(BaseModel):
    is_training: bool = Field(..., description="Whether training is in progress")
    status: str = Field(..., description="Current training status message")


class TrainingStartResponse(BaseModel):
    status: str = Field(..., description="Response status")
    message: str = Field(..., description="Detailed response message")


class WebSocketStatsMessage(BaseModel):
    type: str = Field("stats", description="Message type")
    camera_id: str = Field(..., description="Camera identifier")
    fps: float = Field(..., description="Current frames per second")
    person_count: int = Field(..., description="Number of persons detected")
    male_count: int = Field(..., description="Number of males detected")
    female_count: int = Field(..., description="Number of females detected")
    timestamp: str = Field(..., description="ISO timestamp")
    detections: List[Dict[str, Any]] = Field(..., description="List of detections")

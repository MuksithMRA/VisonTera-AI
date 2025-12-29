from pydantic import BaseModel, Field
from typing import Optional

class DetectionStatus(BaseModel):
    status: str = Field(..., description="System status indicator", example="ok")
    running: bool = Field(..., description="Whether detection is currently running", example=True)
    frames_processed: int = Field(..., description="Total number of frames processed since start", example=1234)
    camera_index: int = Field(..., description="Currently active camera index", example=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "running": True,
                "frames_processed": 1234,
                "camera_index": 0
            }
        }
    }

class StartResponse(BaseModel):
    status: str = Field(..., description="Response status: 'started', 'already_running', or 'error'", example="started")
    message: Optional[str] = Field(None, description="Additional message for error responses", example=None)

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "started",
                "message": None
            }
        }
    }

class StopResponse(BaseModel):
    status: str = Field(..., description="Response status indicating detection stopped", example="stopped")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "stopped"
            }
        }
    }

class TrainingStatus(BaseModel):
    is_training: bool = Field(..., description="Whether training is currently in progress", example=False)
    status: str = Field(..., description="Current training status message", example="idle")

    model_config = {
        "json_schema_extra": {
            "example": {
                "is_training": False,
                "status": "idle"
            }
        }
    }

class TrainingStartResponse(BaseModel):
    status: str = Field(..., description="Response status: 'started' or 'error'", example="started")
    message: str = Field(..., description="Detailed response message", example="Training pipeline started in background.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "started",
                "message": "Training pipeline started in background. Monitor logs or /api/training_status for progress."
            }
        }
    }

class WebSocketStatsMessage(BaseModel):
    fps: float = Field(..., description="Current frames per second", example=30.5)
    person_count: int = Field(..., description="Number of persons detected in current frame", example=3)
    timestamp: str = Field(..., description="ISO timestamp of the stats update", example="2025-12-29T13:45:00.000Z")
    detections: list = Field(..., description="List of detection coordinates for each person", example=[{"x": 100, "y": 150, "width": 80, "height": 200}])

    model_config = {
        "json_schema_extra": {
            "example": {
                "fps": 30.5,
                "person_count": 3,
                "timestamp": "2025-12-29T13:45:00.000Z",
                "detections": [
                    {"x": 100, "y": 150, "width": 80, "height": 200},
                    {"x": 300, "y": 120, "width": 75, "height": 190}
                ]
            }
        }
    }

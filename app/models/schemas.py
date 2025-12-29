from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class DetectionStatus(BaseModel):
    status: str
    running: bool
    frames_processed: int
    camera_index: int

class StartResponse(BaseModel):
    status: str
    message: Optional[str] = None

class StopResponse(BaseModel):
    status: str

class TrainingStatus(BaseModel):
    is_training: bool
    status: str

class TrainingStartResponse(BaseModel):
    status: str
    message: str

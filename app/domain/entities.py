from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
from datetime import datetime


class CameraType(Enum):
    LOCAL = "local"
    RTSP = "rtsp"


class CameraState(Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    RECONNECTING = "reconnecting"


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def bottom_center(self) -> tuple:
        return (self.center_x, self.y2)

    def to_dict(self) -> Dict[str, float]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2
        }


@dataclass
class Detection:
    track_id: int
    bbox: BoundingBox
    confidence: float
    gender: Optional[str] = None
    gender_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.track_id,
            "x": self.bbox.center_x,
            "y": self.bbox.y2,
            "confidence": self.confidence,
            "gender": self.gender or "Person",
            "bbox": self.bbox.to_dict()
        }


@dataclass
class CameraConfig:
    camera_id: str
    source: str
    name: str
    camera_type: CameraType = CameraType.LOCAL
    frame_width: int = 640
    frame_height: int = 480
    target_fps: float = 30.0
    confidence_threshold: float = 0.5
    show_coords: bool = True
    show_fps: bool = True
    box_color: tuple = (0, 255, 136)


@dataclass
class ProcessingStats:
    camera_id: str
    fps: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    person_count: int = 0
    male_count: int = 0
    female_count: int = 0
    detections: list = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "stats",
            "camera_id": self.camera_id,
            "fps": self.fps,
            "width": self.frame_width,
            "height": self.frame_height,
            "person_count": self.person_count,
            "male_count": self.male_count,
            "female_count": self.female_count,
            "detections": self.detections,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class CameraStatus:
    camera_id: str
    state: CameraState
    config: CameraConfig
    stats: Optional[ProcessingStats] = None
    error_message: Optional[str] = None
    frames_processed: int = 0
    last_frame_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "state": self.state.value,
            "name": self.config.name,
            "source": self.config.source,
            "camera_type": self.config.camera_type.value,
            "frames_processed": self.frames_processed,
            "error_message": self.error_message,
            "stats": self.stats.to_dict() if self.stats else None,
            "last_frame_time": self.last_frame_time.isoformat() if self.last_frame_time else None
        }

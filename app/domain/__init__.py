from app.domain.entities import Detection, BoundingBox, CameraConfig, CameraStatus, ProcessingStats
from app.domain.interfaces import IInferenceEngine, ICameraCapture, ICameraProcessor

__all__ = [
    "Detection",
    "BoundingBox", 
    "CameraConfig",
    "CameraStatus",
    "ProcessingStats",
    "IInferenceEngine",
    "ICameraCapture",
    "ICameraProcessor",
]

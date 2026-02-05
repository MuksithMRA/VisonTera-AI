from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np
from app.domain.entities import Detection, CameraConfig, CameraStatus, ProcessingStats


class IInferenceEngine(ABC):
    @abstractmethod
    def load_models(self, detection_model_path: str, gender_model_path: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def detect_persons(self, frame: np.ndarray, confidence: float, camera_id: str) -> List[Detection]:
        pass

    @abstractmethod
    def classify_gender(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Tuple[Optional[str], float]:
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        pass


class ICameraCapture(ABC):
    @abstractmethod
    def open(self, source: str, frame_width: int, frame_height: int) -> bool:
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        pass

    @abstractmethod
    def release(self) -> None:
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        pass


class ICameraProcessor(ABC):
    @abstractmethod
    async def start(self) -> bool:
        pass

    @abstractmethod
    async def stop(self) -> None:
        pass

    @abstractmethod
    def get_status(self) -> CameraStatus:
        pass

    @abstractmethod
    def get_latest_frame(self) -> Optional[bytes]:
        pass

    @abstractmethod
    def get_latest_stats(self) -> Optional[ProcessingStats]:
        pass

    @abstractmethod
    def update_config(self, config: CameraConfig) -> None:
        pass

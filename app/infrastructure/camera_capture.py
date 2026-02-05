import cv2
import os
import threading
from typing import Optional, Tuple
import numpy as np
from app.domain.interfaces import ICameraCapture
from app.config import logger


class CameraCapture(ICameraCapture):
    def __init__(self):
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._source: Optional[str] = None

    def open(self, source: str, frame_width: int = 640, frame_height: int = 480) -> bool:
        with self._lock:
            try:
                if self._cap is not None:
                    self._cap.release()

                self._source = source

                if isinstance(source, str) and source.startswith("rtsp://"):
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                    self._cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                else:
                    backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
                    cam_idx = int(source) if isinstance(source, str) and source.isdigit() else source
                    if isinstance(cam_idx, str):
                        cam_idx = 0
                    self._cap = cv2.VideoCapture(cam_idx, backend)

                if self._cap.isOpened():
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
                    self._cap.set(cv2.CAP_PROP_FPS, 30)
                    logger.info(f"Camera opened: {source}")
                    return True

                logger.error(f"Failed to open camera: {source}")
                return False

            except Exception as e:
                logger.error(f"Error opening camera {source}: {e}")
                return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return False, None
            return self._cap.read()

    def release(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
                logger.info(f"Camera released: {self._source}")

    def is_opened(self) -> bool:
        with self._lock:
            return self._cap is not None and self._cap.isOpened()

    def get_frame_dimensions(self) -> Tuple[int, int]:
        with self._lock:
            if self._cap is None:
                return 0, 0
            return (
                int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )

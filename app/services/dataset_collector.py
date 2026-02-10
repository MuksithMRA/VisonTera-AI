import cv2
import threading
import time
import concurrent.futures
from typing import List, Set, Dict, Optional
from pathlib import Path
from datetime import datetime
import numpy as np

from app.config import AppConfig, logger
from app.domain.entities import Detection

class DatasetCollector:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._captured_ids = {}
                    instance._last_capture_time = {}
                    
                    try:
                        instance._output_dir = Path(AppConfig.DATASET_OUTPUT_DIR)
                        instance._output_dir.mkdir(parents=True, exist_ok=True)
                        logger.info(f"DatasetCollector initialized. Output dir: {instance._output_dir}")
                    except Exception as e:
                        logger.error(f"Failed to create dataset output directory: {e}")
                        
                    instance._save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
                    cls._instance = instance
        return cls._instance

    def __init__(self):
        pass

    def process_frame(self, camera_id: str, frame: np.ndarray, detections: List[Detection]) -> None:
        """
        Process frame and detections to capture unique person crops.
        Runs in the main processing thread, but offloads file I/O to a thread pool.
        """
        if not AppConfig.DATASET_COLLECTION_ENABLED:
            return

        current_time = time.time()
        last_time = self._last_capture_time.get(camera_id, 0)

        # Check interval
        if current_time - last_time < AppConfig.DATASET_CAPTURE_INTERVAL:
            return

        # Initialize tracking for verify camera
        if camera_id not in self._captured_ids:
            self._captured_ids[camera_id] = set()

        processed_any = False
        
        for det in detections:
            track_id = det.track_id
            
            # Skip if no track ID or already captured
            if track_id == -1 or track_id in self._captured_ids[camera_id]:
                continue

            # Check dimensions
            bbox = det.bbox
            w = bbox.x2 - bbox.x1
            h = bbox.y2 - bbox.y1

            if h < AppConfig.DATASET_MIN_HEIGHT or w < AppConfig.DATASET_MIN_WIDTH:
                continue

            # Mark as captured for this camera
            self._captured_ids[camera_id].add(track_id)
            processed_any = True
            
            # Submit save task to avoid blocking main thread
            # We must copy the frame because it might be modified or reused in the main loop
            self._save_executor.submit(
                self._save_crop, 
                frame.copy(), 
                det, 
                camera_id
            )

        if processed_any:
            self._last_capture_time[camera_id] = current_time

    def _save_crop(self, frame: np.ndarray, detection: Detection, camera_id: str) -> None:
        try:
            bbox = detection.bbox
            x1, y1 = int(bbox.x1), int(bbox.y1)
            x2, y2 = int(bbox.x2), int(bbox.y2)
            
            # Clamp coordinates to image boundaries
            h_img, w_img = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_img, x2)
            y2 = min(h_img, y2)
            
            if x2 <= x1 or y2 <= y1:
                return

            person_crop = frame[y1:y2, x1:x2]
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Clean filename
            safe_cam_id = "".join(x for x in str(camera_id) if x.isalnum() or x in ('_', '-'))
            filename = f"cam{safe_cam_id}_{timestamp}_id{detection.track_id}.jpg"
            filepath = self._output_dir / filename
            
            cv2.imwrite(str(filepath), person_crop)
            # Log at debug level to avoid spamming
            # logger.debug(f"Captured dataset image: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving dataset crop: {e}")

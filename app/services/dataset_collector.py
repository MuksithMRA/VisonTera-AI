import cv2
import threading
import time
import shutil
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
                    instance._labeling_active = True
                    instance._labeling_thread = None
                    instance._start_labeling_thread()
                    
                    cls._instance = instance
        return cls._instance

    def _start_labeling_thread(self):
        if not AppConfig.DATASET_AUTOLABEL_ENABLED:
            return
            
        if self._labeling_thread and self._labeling_thread.is_alive():
            return
            
        self._labeling_thread = threading.Thread(target=self._auto_label_loop, daemon=True)
        self._labeling_thread.start()
        logger.info("Dataset auto-labeling background thread started")

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

    def _auto_label_loop(self):
        """Background loop to auto-label collected images."""
        from app.infrastructure.inference_engine import InferenceEngine
        inference_engine = InferenceEngine()
        
        # Create directories
        male_dir = self._output_dir / "MALE"
        female_dir = self._output_dir / "FEMALE"
        uncertain_dir = self._output_dir / "UNCERTAIN"
        
        try:
            male_dir.mkdir(parents=True, exist_ok=True)
            female_dir.mkdir(parents=True, exist_ok=True)
            uncertain_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create labeling directories: {e}")
            return
            
        logger.info(f"Auto-labeling loop running. Interval: {AppConfig.DATASET_AUTOLABEL_INTERVAL}s")
        
        while self._labeling_active:
            try:
                time.sleep(AppConfig.DATASET_AUTOLABEL_INTERVAL)
                
                # Check if inference engine is ready
                if not inference_engine.is_loaded():
                    continue
                    
                # Scan for images in the root output directory (not recursive)
                images = [f for f in self._output_dir.iterdir() 
                          if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                
                if not images:
                    continue
                    
                logger.debug(f"Auto-labeling found {len(images)} images to process")
                
                for img_path in images:
                    try:
                        # Read image
                        img = cv2.imread(str(img_path))
                        if img is None:
                            logger.warning(f"Failed to read image for labeling: {img_path}")
                            continue
                            
                        # Predict gender
                        gender, confidence = inference_engine.predict_gender_from_crop(img)
                        
                        target_dir = uncertain_dir
                        if gender and confidence >= AppConfig.DATASET_AUTOLABEL_CONFIDENCE:
                            if gender == "Male":
                                target_dir = male_dir
                            elif gender == "Female":
                                target_dir = female_dir
                        
                        # Move file
                        dest_path = target_dir / img_path.name
                        shutil.move(str(img_path), str(dest_path))
                        
                        # logger.debug(f"Labeled {img_path.name} -> {target_dir.name} ({confidence:.2f})")
                        
                    except Exception as e:
                        logger.error(f"Error processing image {img_path}: {e}")
                        
            except Exception as e:
                logger.error(f"Error in auto-labeling loop: {e}")
                time.sleep(5) # Wait a bit on error

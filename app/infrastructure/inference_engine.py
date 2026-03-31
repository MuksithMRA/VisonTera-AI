import torch
import torch.nn as nn
import cv2
import numpy as np
import threading
from typing import Any, List, Optional, Tuple, Dict
from pathlib import Path
from ultralytics import YOLO
from torchvision import models
from app.domain.interfaces import IInferenceEngine
from app.domain.entities import Detection, BoundingBox
from app.config import AppConfig, logger
from app.infrastructure.reid_extractor import ReIDFeatureExtractor
from app.infrastructure.cross_camera_reid import CrossCameraReIDManager
from app.infrastructure.tracking_quality import TrackingQualityMonitor


class ResNet50GenderClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class InferenceEngine(IInferenceEngine):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._detection_model: Optional[YOLO] = None
        self._detection_model_path: Optional[str] = None
        self._gender_model = None
        self._gender_model_type: Optional[str] = None
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._use_half = self._device == 'cuda'
        self._inference_lock = threading.Lock()
        self._track_histories: Dict[str, Dict[int, dict]] = {}
        self._gender_classes = ['Female', 'Male']
        self._gender_check_interval = AppConfig.GENDER_CHECK_INTERVAL
        self._gender_voting_enabled = AppConfig.GENDER_VOTING_ENABLED
        self._gender_vote_threshold = AppConfig.GENDER_VOTE_THRESHOLD
        
        self._uniform_model = None
        self._uniform_filter_enabled = AppConfig.UNIFORM_FILTER_ENABLED
        self._uniform_check_interval = AppConfig.UNIFORM_CHECK_INTERVAL
        self._uniform_voting_enabled = AppConfig.UNIFORM_VOTING_ENABLED
        self._uniform_vote_threshold = AppConfig.UNIFORM_VOTE_THRESHOLD
        self._uniform_confidence_threshold = AppConfig.UNIFORM_CONFIDENCE_THRESHOLD
        
        self._frame_counts: Dict[str, int] = {}
        _tp = AppConfig.TRACKER_CONFIG_PATH
        self._tracker_config = str(_tp.resolve()) if _tp.exists() else "botsort.yaml"
        self._reid_extractor = ReIDFeatureExtractor(
            device=self._device,
            use_half=self._use_half
        )
        self._reid_manager = CrossCameraReIDManager(
            similarity_threshold=AppConfig.REID_SIMILARITY_THRESHOLD,
            max_gallery_size=AppConfig.REID_MAX_GALLERY_SIZE,
            stale_timeout=AppConfig.REID_STALE_TIMEOUT_SEC,
            gallery_pollution_min_sim=AppConfig.REID_GALLERY_POLLUTION_SIM,
        )
        self._reid_refresh_interval = AppConfig.REID_REFRESH_INTERVAL_FRAMES
        self._reid_min_crop_height = AppConfig.REID_MIN_CROP_HEIGHT
        self._reid_min_crop_width = AppConfig.REID_MIN_CROP_WIDTH
        self._reid_conf_threshold = AppConfig.REID_MIN_CONFIDENCE
        self._reid_merge_every = AppConfig.REID_MERGE_EVERY_FRAMES
        self._reid_cleanup_every = AppConfig.REID_CLEANUP_EVERY_FRAMES
        self._tracking_quality = TrackingQualityMonitor(
            fragmentation_iou=AppConfig.FRAGMENTATION_IOU_THRESHOLD,
            collision_iou_max=AppConfig.COLLISION_IOU_MAX,
            ghost_unassigned_frames=AppConfig.GHOST_UNASSIGNED_FRAMES,
        )
        # Ultralytics keeps one predictor.trackers[0] for numpy/frame calls. Interleaving two cameras
        # reuses the same BoT-SORT as if frames were one stream — corrupting IDs and Re-ID keys.
        self._per_camera_ultralytics_trackers: Dict[str, Any] = {}
        self._tracker_active_slot: Optional[str] = None

        logger.info(f"InferenceEngine initialized: device={self._device}, half={self._use_half}, CUDA available={torch.cuda.is_available()}")

    def _make_new_botsort_tracker(self) -> Any:
        """Construct a BoT-SORT / ByteTrack instance matching TRACKER_CONFIG_PATH (same as Ultralytics on_predict_start)."""
        from ultralytics.trackers.track import TRACKER_MAP, check_yaml
        from ultralytics.utils import YAML, IterableSimpleNamespace

        tracker_path = check_yaml(self._tracker_config)
        cfg = IterableSimpleNamespace(**YAML.load(tracker_path))
        if cfg.tracker_type not in TRACKER_MAP:
            raise AssertionError(f"Unsupported tracker type: {cfg.tracker_type}")
        return TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)

    def _swap_ultralytics_tracker_for_camera(self, camera_id: str) -> None:
        """Before model.track(), install this camera's BoT-SORT so streams do not share one tracker state."""
        if self._detection_model is None:
            return
        pred = getattr(self._detection_model, "predictor", None)
        if pred is None or not getattr(pred, "trackers", None):
            return

        if self._tracker_active_slot is not None:
            self._per_camera_ultralytics_trackers[self._tracker_active_slot] = pred.trackers[0]

        if camera_id not in self._per_camera_ultralytics_trackers:
            self._per_camera_ultralytics_trackers[camera_id] = self._make_new_botsort_tracker()

        pred.trackers[0] = self._per_camera_ultralytics_trackers[camera_id]
        self._tracker_active_slot = camera_id

    def _sync_ultralytics_tracker_cache(self, camera_id: str) -> None:
        """After model.track(), remember which BoT-SORT instance is live (updated in place)."""
        if self._detection_model is None:
            return
        pred = getattr(self._detection_model, "predictor", None)
        if pred is None or not getattr(pred, "trackers", None):
            return
        self._per_camera_ultralytics_trackers[camera_id] = pred.trackers[0]
        self._tracker_active_slot = camera_id

    def _get_latest_versioned_model(self, prefix: str) -> Optional[str]:
        """Find the latest versioned model directory matching the given prefix.
        
        Args:
            prefix: Directory name prefix to match (e.g., 'v_' for gender, 'crowd_v_' for detection)
        
        Returns:
            Path to best model weights, or None if not found.
        """
        try:
            models_dir = Path("infrastructure/models")
            if not models_dir.exists():
                return None
            version_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
            version_dirs.sort(key=lambda x: x.name, reverse=True)
            for v_dir in version_dirs:
                # Check both naming conventions: best_model.pt (our copy) and weights/best.pt (YOLO default)
                for candidate in [v_dir / "best_model.pt", v_dir / "weights" / "best.pt"]:
                    if candidate.exists():
                        logger.info(f"Found latest {prefix} model: {v_dir.name} -> {candidate}")
                        return str(candidate)
            return None
        except Exception as e:
            logger.error(f"Error searching for latest {prefix} model: {e}")
            return None

    def _get_latest_model_path(self) -> Optional[str]:
        """Find latest gender classification model (v_* directories)."""
        return self._get_latest_versioned_model("v_")

    def _get_latest_detection_model_path(self) -> Optional[str]:
        """Find latest crowd detection model (crowd_v_* directories)."""
        return self._get_latest_versioned_model("crowd_v_")

    def get_available_detection_models(self) -> list:
        """Scan for all available detection models (.pt files) across known directories."""
        models = []
        search_dirs = [
            Path("infrastructure/models"),
            Path("scut_head/models"),
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            # Top-level .pt files (e.g., yolo26m.pt, yolo26m-head.pt)
            for pt_file in search_dir.glob("*.pt"):
                # Skip gender/classification models
                if any(skip in pt_file.name for skip in ['gender', 'cls', 'peta', 'face']):
                    continue
                models.append({
                    'name': pt_file.stem,
                    'path': str(pt_file),
                    'type': 'base',
                    'active': str(pt_file) == self._detection_model_path
                })

            # Versioned directories (e.g., crowd_v_*/best_model.pt, scut_head_*/best_model.pt)
            for v_dir in sorted(search_dir.iterdir(), key=lambda x: x.name, reverse=True):
                if not v_dir.is_dir():
                    continue
                for candidate in [v_dir / "best_model.pt", v_dir / "weights" / "best.pt"]:
                    if candidate.exists():
                        models.append({
                            'name': v_dir.name,
                            'path': str(candidate),
                            'type': 'trained',
                            'active': str(candidate) == self._detection_model_path
                        })
                        break  # Only use best_model.pt, not both

        return models

    def switch_detection_model(self, model_path: str) -> dict:
        """Hot-swap the detection model without restarting the server."""
        try:
            path = Path(model_path)
            if not path.exists():
                return {'status': 'error', 'message': f'Model not found: {model_path}'}

            logger.info(f"Switching detection model to: {model_path}")

            # Load the new model
            new_model = YOLO(model_path)
            if self._device == 'cuda':
                new_model.to(self._device)

            # Swap with lock to avoid inference during swap
            with self._inference_lock:
                self._detection_model = new_model
                self._detection_model_path = str(path)
                self._track_histories.clear()
                self._frame_counts.clear()
                self._tracking_quality.reset()
                self._per_camera_ultralytics_trackers.clear()
                self._tracker_active_slot = None

            logger.info(f"Detection model switched successfully to: {path.name}")
            return {
                'status': 'success',
                'model': path.name,
                'path': str(path),
                'device': self._device
            }

        except Exception as e:
            logger.error(f"Failed to switch detection model: {e}")
            return {'status': 'error', 'message': str(e)}

    @property
    def current_detection_model_path(self) -> Optional[str]:
        return self._detection_model_path

    def load_models(self, detection_model_path: str, gender_model_path: Optional[str] = None) -> None:
        try:
            # ── Detection Model: prefer latest crowd-trained model ──
            crowd_model = self._get_latest_detection_model_path()
            if crowd_model:
                detection_model_path = crowd_model
                logger.info(f"Using crowd-optimized detection model: {crowd_model}")
            else:
                logger.info(f"No crowd model found, using default: {detection_model_path}")

            self._detection_model = YOLO(detection_model_path)
            self._detection_model_path = detection_model_path
            if self._device == 'cuda':
                self._detection_model.to(self._device)
                logger.info(f"Detection model loaded on GPU: {torch.cuda.get_device_name(0)}, half={self._use_half}")
            else:
                logger.warning(f"Detection model loaded on CPU - inference will be slow!")

            # ── Gender Model: prefer latest versioned model ──
            gender_path = self._get_latest_model_path()
            if not gender_path:
                default_path = "infrastructure/models/resnet50-gender.pt"
                if Path(default_path).exists():
                    gender_path = default_path

            if not gender_path and gender_model_path:
                gender_path = gender_model_path

            if gender_path and Path(gender_path).exists():
                self._load_gender_model(gender_path)
            else:
                logger.warning("Gender model not found. Gender detection disabled.")

            # ── Uniform Model (Employee Exclusion) ──
            if self._uniform_filter_enabled:
                self._load_uniform_model()

            # ── Re-ID Model: FastReID SBS for cross-camera deduplication ──
            reid_loaded = self._reid_extractor.load()
            if reid_loaded:
                logger.info("Cross-camera Re-ID feature extractor loaded successfully")
            else:
                logger.warning(
                    "Re-ID model failed to load. Cross-camera deduplication disabled. "
                    "Person counts will be summed per camera (may include duplicates)."
                )

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def _load_gender_model(self, model_path: str) -> None:
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            is_yolo = isinstance(checkpoint, dict) and 'model' in checkpoint

            if is_yolo:
                self._gender_model = YOLO(model_path)
                self._gender_model_type = 'yolo'
                logger.info("YOLO-cls gender classification model loaded.")
            else:
                self._gender_model = ResNet50GenderClassifier(num_classes=2)
                self._gender_model.load_state_dict(checkpoint)
                self._gender_model.to(self._device)
                self._gender_model.eval()
                self._gender_model_type = 'resnet50'
                logger.info("ResNet50 gender classification model loaded.")
        except Exception as e:
            logger.error(f"Error loading gender model: {e}")
            self._gender_model = None

    def _load_uniform_model(self) -> None:
        uniform_model_path = AppConfig.UNIFORM_MODEL_PATH
        models_dir = Path("infrastructure/models")
        if models_dir.exists():
            uniform_dirs = sorted(
                [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("uniform_v_")],
                key=lambda x: x.name, reverse=True
            )
            for u_dir in uniform_dirs:
                candidate = u_dir / "best_model.pt"
                if candidate.exists():
                    uniform_model_path = str(candidate)
                    logger.info(f"Found latest uniform model version: {u_dir.name}")
                    break

        if Path(uniform_model_path).exists():
            try:
                self._uniform_model = YOLO(uniform_model_path)
                if self._device == 'cuda':
                    self._uniform_model.to(self._device)
                logger.info(f"Uniform detector loaded: {uniform_model_path}")
            except Exception as e:
                logger.error(f"Failed to load uniform classifier: {e}")
                self._uniform_model = None
        else:
            logger.warning(f"Uniform detector not found at {uniform_model_path}. Employee filtering disabled.")
            self._uniform_model = None

    def _predict_uniform(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Tuple[Optional[str], float]:
        if self._uniform_model is None:
            return None, 0.0

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None, 0.0

        person_img = frame[y1:y2, x1:x2]

        try:
            with self._inference_lock:
                results = self._uniform_model.predict(
                    person_img, 
                    verbose=False, 
                    imgsz=640,
                    conf=self._uniform_confidence_threshold,
                    device=self._device,
                    half=self._use_half
                )
            
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    max_conf = float(boxes.conf.max())
                    return 'employee', max_conf
            
            return 'visitor', 0.0
        except Exception as e:
            logger.error(f"Uniform prediction error: {e}")
        return None, 0.0

    def detect_persons(self, frame: np.ndarray, confidence: float, camera_id: str) -> List[Detection]:
        if self._detection_model is None:
            return []

        if camera_id not in self._track_histories:
            self._track_histories[camera_id] = {}
        if camera_id not in self._frame_counts:
            self._frame_counts[camera_id] = 0

        self._frame_counts[camera_id] += 1
        track_history = self._track_histories[camera_id]
        frame_count = self._frame_counts[camera_id]

        with self._inference_lock:
            self._swap_ultralytics_tracker_for_camera(camera_id)
            results = self._detection_model.track(
                frame,
                classes=[0],
                conf=confidence,
                persist=True,
                verbose=False,
                tracker=self._tracker_config,
                device=self._device,
                half=self._use_half
            )
            self._sync_ultralytics_tracker_cache(camera_id)

        detections = []
        if results and results[0].boxes:
            boxes = results[0].boxes
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
            
            # Inform Re-ID manager about currently active tracks on this camera
            if ids is not None:
                self._reid_manager.update_active_tracks(camera_id, ids.tolist())

            # Prepare for overlap check (occlusion filtering)
            all_boxes = boxes.xyxy.cpu().numpy()
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                track_id = int(ids[i]) if ids is not None else -1

                gender_label = None
                if track_id != -1:
                    if track_id not in track_history:
                        track_history[track_id] = {
                            'gender': None,
                            'gender_confidence': 0.0,
                            'male_votes': 0,
                            'female_votes': 0,
                            'last_check': 0,
                            'uniform_label': None,
                            'uniform_confidence': 0.0,
                            'employee_votes': 0,
                            'visitor_votes': 0,
                            'uniform_last_check': 0,
                            'is_employee': False,
                            'reid_last_extract': 0,
                        }

                    hist = track_history[track_id]
                    should_check = (frame_count - hist.get('last_check', 0)) > self._gender_check_interval
                    if hist['gender'] is None:
                        should_check = True

                    if should_check and self._gender_model is not None:
                        gender, gender_conf = self.classify_gender(frame, (x1, y1, x2, y2))
                        hist['last_check'] = frame_count

                        if gender:
                            if self._gender_voting_enabled:
                                if gender == 'Male':
                                    hist['male_votes'] += 1
                                else:
                                    hist['female_votes'] += 1

                                if hist['male_votes'] >= self._gender_vote_threshold or \
                                   hist['female_votes'] >= self._gender_vote_threshold:
                                    hist['gender'] = 'Male' if hist['male_votes'] > hist['female_votes'] else 'Female'
                                    hist['gender_confidence'] = gender_conf
                            else:
                                hist['gender'] = gender
                                hist['gender_confidence'] = gender_conf

                    gender_label = hist['gender']

                    # ── Uniform Classification (Employee Exclusion) ──
                    if self._uniform_filter_enabled and self._uniform_model is not None:
                        uniform_should_check = (frame_count - hist.get('uniform_last_check', 0)) > self._uniform_check_interval
                        if hist['uniform_label'] is None:
                            uniform_should_check = True

                        if uniform_should_check:
                            uniform_label, uniform_conf = self._predict_uniform(frame, (x1, y1, x2, y2))
                            hist['uniform_last_check'] = frame_count

                            if uniform_label:
                                if self._uniform_voting_enabled:
                                    if uniform_label == 'employee':
                                        hist['employee_votes'] += 1
                                    else:
                                        hist['visitor_votes'] += 1

                                    if (hist['employee_votes'] >= self._uniform_vote_threshold or
                                            hist['visitor_votes'] >= self._uniform_vote_threshold):
                                        hist['uniform_label'] = 'employee' if hist['employee_votes'] > hist['visitor_votes'] else 'visitor'
                                        hist['uniform_confidence'] = uniform_conf
                                        hist['is_employee'] = (hist['uniform_label'] == 'employee')
                                else:
                                    hist['uniform_label'] = uniform_label
                                    hist['uniform_confidence'] = uniform_conf
                                    hist['is_employee'] = (uniform_label == 'employee')

                    is_employee_val = hist.get('is_employee', False)

                # ── Cross-Camera Re-ID ──
                global_id = -1
                if track_id != -1 and self._reid_extractor.is_loaded:
                    overlap_detected = False
                    for j in range(len(all_boxes)):
                        if i == j: continue
                        # Calculate IoU/Overlap area
                        b1 = all_boxes[i]
                        b2 = all_boxes[j]
                        # Intersection
                        ix1 = max(b1[0], b2[0])
                        iy1 = max(b1[1], b2[1])
                        ix2 = min(b1[2], b2[2])
                        iy2 = min(b1[3], b2[3])
                        if ix2 > ix1 and iy2 > iy1:
                            intersection = (ix2 - ix1) * (iy2 - iy1)
                            b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
                            # If box is > 30% covered by another box, skip Re-ID
                            if intersection / b1_area > AppConfig.FRAGMENTATION_IOU_THRESHOLD:
                                overlap_detected = True
                                break

                    should_extract_reid = (
                        (hist.get('reid_last_extract', 0) == 0 or
                        (frame_count - hist.get('reid_last_extract', 0)) > self._reid_refresh_interval)
                        and conf >= self._reid_conf_threshold
                        and not overlap_detected
                    )

                    if should_extract_reid:
                        global_id = self._extract_and_assign_global_id(
                            frame, x1, y1, x2, y2,
                            camera_id, track_id, gender_label, is_employee_val
                        )
                        # Only advance the refresh interval after a successful assignment.
                        # If extraction failed (crop too small, embedding error), retry next frame
                        # instead of waiting REID_REFRESH_INTERVAL_FRAMES with global_id stuck at -1.
                        if global_id >= 0:
                            hist['reid_last_extract'] = frame_count
                    else:
                        cached_gid = self._reid_manager.get_global_id(camera_id, track_id)
                        if cached_gid is not None:
                            global_id = cached_gid
                            self._reid_manager.touch_person(cached_gid)
                        elif conf < self._reid_conf_threshold or overlap_detected:
                            hist['reid_last_extract'] = max(0, frame_count - (self._reid_refresh_interval // 2))

                detection = Detection(
                    track_id=track_id,
                    bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                    confidence=conf,
                    global_id=global_id,
                    gender=gender_label,
                    class_id=int(box.cls[0].cpu().numpy()),
                    is_employee=is_employee_val if track_id != -1 else False
                )
                detections.append(detection)

        qlist = []
        for det in detections:
            bb = det.bbox
            qlist.append((
                det.track_id,
                np.array([bb.x1, bb.y1, bb.x2, bb.y2], dtype=np.float32),
                det.global_id,
            ))
        self._tracking_quality.process_frame(camera_id, qlist)

        if frame_count % self._reid_merge_every == 0:
            self._reid_manager.merge_duplicates()
        if frame_count % self._reid_cleanup_every == 0:
            self._reid_manager.cleanup_stale()

        return detections

    def classify_gender(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Tuple[Optional[str], float]:
        if self._gender_model is None:
            return None, 0.0

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None, 0.0

        person_img = frame[y1:y2, x1:x2]

        try:
            if self._gender_model_type == 'yolo':
                with self._inference_lock:
                    results = self._gender_model.predict(person_img, verbose=False, imgsz=224, device=self._device, half=self._use_half)
                if results and len(results) > 0:
                    probs = results[0].probs
                    if probs is not None:
                        predicted = probs.top1
                        confidence = float(probs.top1conf)
                        if confidence < 0.6:
                            return None, confidence
                        return self._gender_classes[predicted], confidence
            else:
                img = cv2.resize(person_img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.transpose((2, 0, 1))
                img = np.ascontiguousarray(img, dtype=np.float32)
                img /= 255.0
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                img = (img - mean) / std

                img_tensor = torch.from_numpy(img).unsqueeze(0).to(self._device).float()

                with torch.no_grad():
                    outputs = self._gender_model(img_tensor)
                    if isinstance(outputs, (list, tuple)):
                        outputs = outputs[0]
                    probs = torch.softmax(outputs, dim=1).squeeze()

                confidence, predicted = torch.max(probs, 0)
                if confidence.item() < 0.6:
                    return None, confidence.item()
                return self._gender_classes[predicted.item()], confidence.item()

        except Exception as e:
            logger.error(f"Gender prediction error: {e}")

        return None, 0.0

    def predict_gender_from_crop(self, crop: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Predict gender from a cropped image.
        """
        if self._gender_model is None:
            return None, 0.0
            
        try:
            if self._gender_model_type == 'yolo':
                 with self._inference_lock:
                    results = self._gender_model.predict(crop, verbose=False, imgsz=224, device=self._device, half=self._use_half)
                 if results and len(results) > 0:
                    probs = results[0].probs
                    if probs is not None:
                        predicted = probs.top1
                        confidence = float(probs.top1conf)
                        return self._gender_classes[predicted], confidence
            else:
                img = cv2.resize(crop, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.transpose((2, 0, 1))
                img = np.ascontiguousarray(img, dtype=np.float32)
                img /= 255.0
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                img = (img - mean) / std

                img_tensor = torch.from_numpy(img).unsqueeze(0).to(self._device).float()

                with torch.no_grad():
                    outputs = self._gender_model(img_tensor)
                    if isinstance(outputs, (list, tuple)):
                        outputs = outputs[0]
                    probs = torch.softmax(outputs, dim=1).squeeze()

                confidence, predicted = torch.max(probs, 0)
                return self._gender_classes[predicted.item()], confidence.item()

        except Exception as e:
            logger.error(f"Gender crop prediction error: {e}")
            
        return None, 0.0

    def is_loaded(self) -> bool:
        return self._detection_model is not None

    def clear_track_history(self, camera_id: str) -> None:
        if camera_id in self._track_histories:
            del self._track_histories[camera_id]
        if camera_id in self._frame_counts:
            del self._frame_counts[camera_id]
        self._per_camera_ultralytics_trackers.pop(camera_id, None)
        if self._tracker_active_slot == camera_id:
            self._tracker_active_slot = None
        self._reid_manager.clear_camera(camera_id)
        self._tracking_quality.prune_camera(camera_id)

    def _extract_and_assign_global_id(
        self,
        frame: np.ndarray,
        x1: float, y1: float, x2: float, y2: float,
        camera_id: str,
        local_track_id: int,
        gender: str = None,
        is_employee: bool = False,
    ) -> int:
        try:
            h, w = frame.shape[:2]
            cx1, cy1 = max(0, int(x1)), max(0, int(y1))
            cx2, cy2 = min(w, int(x2)), min(h, int(y2))

            crop_h = cy2 - cy1
            crop_w = cx2 - cx1

            if crop_h < self._reid_min_crop_height or crop_w < self._reid_min_crop_width:
                cached = self._reid_manager.get_global_id(camera_id, local_track_id)
                return cached if cached is not None else -1

            person_crop = frame[cy1:cy2, cx1:cx2]
            if person_crop.size == 0:
                cached = self._reid_manager.get_global_id(camera_id, local_track_id)
                return cached if cached is not None else -1

            embedding = self._reid_extractor.extract(person_crop)
            if embedding is None:
                cached = self._reid_manager.get_global_id(camera_id, local_track_id)
                return cached if cached is not None else -1

            global_id = self._reid_manager.assign_global_id(
                camera_id=camera_id,
                local_track_id=local_track_id,
                feature_embedding=embedding,
                gender=gender,
                is_employee=is_employee,
            )
            return global_id

        except Exception as e:
            logger.error(f"Re-ID assignment error: {e}")
            cached = self._reid_manager.get_global_id(camera_id, local_track_id)
            return cached if cached is not None else -1

    def get_deduplicated_counts(self) -> Dict:
        return self._reid_manager.get_deduplicated_counts()

    def get_currently_visible_persons(self, max_age: float = 5.0) -> Dict:
        return self._reid_manager.get_currently_visible(max_age=max_age)

    def get_reid_stats(self) -> Dict:
        out = dict(self._reid_manager.get_stats())
        out.update(self._tracking_quality.snapshot())
        out["tracker_backend"] = "botsort"
        out["tracker_config"] = self._tracker_config
        return out

    @property
    def reid_manager(self) -> CrossCameraReIDManager:
        return self._reid_manager

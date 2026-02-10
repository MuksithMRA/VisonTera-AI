import torch
import torch.nn as nn
import cv2
import numpy as np
import threading
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from ultralytics import YOLO
from torchvision import models
from app.domain.interfaces import IInferenceEngine
from app.domain.entities import Detection, BoundingBox
from app.config import AppConfig, logger


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
        self._gender_model = None
        self._gender_model_type: Optional[str] = None
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._inference_lock = threading.Lock()
        self._track_histories: Dict[str, Dict[int, dict]] = {}
        self._gender_classes = ['Female', 'Male']
        self._gender_check_interval = AppConfig.GENDER_CHECK_INTERVAL
        self._gender_voting_enabled = AppConfig.GENDER_VOTING_ENABLED
        self._gender_vote_threshold = AppConfig.GENDER_VOTE_THRESHOLD
        self._frame_counts: Dict[str, int] = {}

    def _get_latest_model_path(self) -> Optional[str]:
        try:
            models_dir = Path("infrastructure/models")
            if not models_dir.exists():
                return None
            version_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("v_")]
            version_dirs.sort(key=lambda x: x.name, reverse=True)
            for v_dir in version_dirs:
                model_path = v_dir / "best_model.pt"
                if model_path.exists():
                    logger.info(f"Found latest model version: {v_dir.name}")
                    return str(model_path)
            return None
        except Exception as e:
            logger.error(f"Error searching for latest model: {e}")
            return None

    def load_models(self, detection_model_path: str, gender_model_path: Optional[str] = None) -> None:
        try:
            self._detection_model = YOLO(detection_model_path)
            self._detection_model.to(self._device)
            logger.info(f"Detection model loaded on device: {self._device}")

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
            results = self._detection_model.track(
                frame,
                classes=[0],
                conf=confidence,
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml"
            )

        detections = []
        if results and results[0].boxes:
            boxes = results[0].boxes
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

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
                            'last_check': 0
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

                detection = Detection(
                    track_id=track_id,
                    bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                    confidence=conf,
                    gender=gender_label
                )
                detections.append(detection)

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
                    results = self._gender_model.predict(person_img, verbose=False, imgsz=224)
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
                    results = self._gender_model.predict(crop, verbose=False, imgsz=224)
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

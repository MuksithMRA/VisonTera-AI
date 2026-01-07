import torch
import torch.nn as nn
import cv2
import numpy as np
from datetime import datetime
import asyncio
from pathlib import Path
import concurrent.futures
import httpx
import os
from ultralytics import YOLO
from torchvision import models
from app.config import AppConfig, logger


class ResNet50GenderClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class DetectionEngine:
    def __init__(self):
        self.model = None
        self.gender_net = None
        self.gender_classes = ['Female', 'Male']
        self.track_history = {}
        self.is_running = False
        self.camera_index = AppConfig.CAMERA_INDEX
        self.frame_count = 0
        self.gender_check_interval = AppConfig.PETA_CHECK_INTERVAL
        self.confidence = AppConfig.CONFIDENCE_THRESHOLD
        self.api_url = AppConfig.API_URL
        self.api_token = AppConfig.API_TOKEN
        self.last_api_push = datetime.min
        self.show_coords = True
        self.show_fps = True
        self.box_color = (0, 255, 136)
        self.frame_queue = asyncio.Queue(maxsize=5)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.stats_queue = asyncio.Queue(maxsize=1)
        self.frame_width = AppConfig.FRAME_WIDTH
        self.frame_height = AppConfig.FRAME_HEIGHT
        self.cap = None
        self.current_frame = None
        self.last_detections = []
        self.capture_task = None
        self.stats_task = None

    def _get_latest_model_path(self):
        """Finds the latest versioned model in infrastructure/models."""
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

    def load_model(self, model_path: str = None):
        if model_path is None: model_path = AppConfig.MODEL_PATH
        try:
            self.model = YOLO(model_path)
            
            gender_model_path = self._get_latest_model_path()
            
            if not gender_model_path:
                default_path = "infrastructure/models/resnet50-gender.pt"
                if Path(default_path).exists():
                    gender_model_path = default_path
                    logger.info("Using default non-versioned ResNet50 model")
            
            old_gender_path = AppConfig.GENDER_MODEL_PATH
            
            if gender_model_path and Path(gender_model_path).exists():
                logger.info(f"Loading Gender Model from: {gender_model_path}")
                try:
                    self.gender_net = ResNet50GenderClassifier(num_classes=2)
                    state_dict = torch.load(gender_model_path)
                    self.gender_net.load_state_dict(state_dict)
                    logger.info("ResNet50 gender classification model loaded successfully.")
                except Exception as e:
                    logger.error(f"Error loading ResNet50 gender model: {e}")
                    self.gender_net = None
            elif Path(old_gender_path).exists():
                try:
                    from ultralytics import YOLO as YOLO_CLS
                    base_model_path = 'infrastructure/models/yolo11m-cls.pt'
                    if not Path(base_model_path).exists():
                        base_model_path = 'yolo11m-cls.pt'
                    
                    base_cls_model = YOLO_CLS(base_model_path)
                    self.gender_net = base_cls_model.model
                    
                    if hasattr(self.gender_net, 'model') and isinstance(self.gender_net.model[-1], nn.Linear):
                        in_features = self.gender_net.model[-1].in_features
                        self.gender_net.model[-1] = nn.Linear(in_features, 2)
                    elif hasattr(self.gender_net.model[-1], 'linear'):
                        in_features = self.gender_net.model[-1].linear.in_features
                        self.gender_net.model[-1].linear = nn.Linear(in_features, 2)
                    
                    state_dict = torch.load(old_gender_path)
                    self.gender_net.load_state_dict(state_dict)
                    logger.info("YOLO gender classification model loaded (fallback).")
                except Exception as e:
                    logger.error(f"Error loading fallback gender model: {e}")
                    self.gender_net = None
            else:
                logger.warning(f"Gender model not found. Gender detection disabled.")
                self.gender_net = None

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(device)
            if self.gender_net:
                self.gender_net.to(device)
                self.gender_net.eval()
                
            logger.info(f"YOLO detections using device: {device}")
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def _predict_gender(self, frame, person_box):
        x1, y1, x2, y2 = [int(v) for v in person_box]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1: return None, 0.0
        
        person_img = frame[y1:y2, x1:x2]
        
        if self.gender_net:
            try:
                img = cv2.resize(person_img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.transpose((2, 0, 1))
                img = np.ascontiguousarray(img, dtype=np.float32)
                img /= 255.0
                img = (img - np.array([0.485, 0.456, 0.406]).reshape(3,1,1)) / np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
                
                img_tensor = torch.from_numpy(img).unsqueeze(0)
                device = next(self.gender_net.parameters()).device
                img_tensor = img_tensor.to(device).float()
                
                with torch.no_grad():
                    outputs = self.gender_net(img_tensor)
                    if isinstance(outputs, (list, tuple)): outputs = outputs[0]
                    probs = torch.softmax(outputs, dim=1).squeeze()
                
                confidence, predicted = torch.max(probs, 0)
                gender = self.gender_classes[predicted.item()]
                
                if confidence.item() < 0.6:
                    return None, confidence.item()
                
                return gender, confidence.item()
            except Exception as e:
                logger.error(f"Gender prediction error: {e}")
        return None, 0.0

    def detect_persons(self, frame):
        if self.model is None: return []
        results = self.model.track(frame, classes=[0], conf=self.confidence, persist=True, verbose=False, tracker="bytetrack.yaml")
        detections = []
        if results and results[0].boxes:
            boxes = results[0].boxes
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                track_id = int(ids[i]) if ids is not None else -1
                if track_id != -1:
                    if track_id not in self.track_history:
                        self.track_history[track_id] = {
                            'gender': None, 
                            'gender_confidence': 0.0,
                            'male_votes': 0, 
                            'female_votes': 0, 
                            'last_check': 0
                        }
                    hist = self.track_history[track_id]
                    
                    should_check = (self.frame_count - hist.get('last_check', 0)) > self.gender_check_interval
                    if hist['gender'] is None: should_check = True
                    
                    if should_check and self.gender_net:
                        gender, gender_conf = self._predict_gender(frame, (x1, y1, x2, y2))
                        hist['last_check'] = self.frame_count
                        
                        if gender:
                            if gender == 'Male':
                                hist['male_votes'] += 1
                            else:
                                hist['female_votes'] += 1
                            
                            if hist['male_votes'] >= 3 or hist['female_votes'] >= 3:
                                hist['gender'] = 'Male' if hist['male_votes'] > hist['female_votes'] else 'Female'
                                hist['gender_confidence'] = gender_conf

                    gender_label = hist['gender'] if hist['gender'] else "Person"
                else:
                    gender_label = "Person"
                
                detections.append({
                    'x': float((x1 + x2) / 2),
                    'y': float(y2),
                    'confidence': conf,
                    'gender': gender_label,
                    'id': track_id,
                    'bbox': {
                        'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)
                    }
                })
        return detections

    def draw_detections(self, frame, detections):
        annotated = frame.copy()
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            conf = det['confidence']
            bottom_x, bottom_y = int(det['x']), int(det['y'])
            gender = det.get('gender', 'Person')
            
            if gender == 'Male':
                box_color = (255, 150, 50)
            elif gender == 'Female':
                box_color = (180, 105, 255)
            else:
                box_color = self.box_color
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            cv2.rectangle(annotated, (x1-1, y1-1), (x2+1, y2+1), (0, 100, 50), 1)
            cv2.circle(annotated, (bottom_x, bottom_y), 8, (0, 212, 255), -1)
            cv2.circle(annotated, (bottom_x, bottom_y), 10, (255, 255, 255), 2)
            
            label = f"{gender} {conf:.0%}"
                
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), box_color, -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            if self.show_coords:
                coord_text = f"({bottom_x}, {bottom_y})"
                cv2.putText(annotated, coord_text, (bottom_x + 15, bottom_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 212, 255), 2)
        return annotated

    async def _push_to_api(self, detections):
        if not self.api_url or not self.api_token:
            return
        url = f"{self.api_url}/agent/stats"
        now = datetime.now()
        now_iso = now.isoformat()
        boxes = []
        male_count = 0
        female_count = 0
        for d in detections:
            g = d.get('gender', 'Person')
            if g == 'Male':
                g_code = 'M'
                male_count += 1
            elif g == 'Female':
                g_code = 'F'
                female_count += 1
            else:
                continue
            boxes.append({
                "camera_id": self.camera_index,
                "date": now_iso,
                "bbox_id": d.get('id', -1),
                "bbox_left": int(d['bbox']['x1']),
                "bbox_top": int(d['bbox']['y1']),
                "bbox_w": int(d['bbox']['x2'] - d['bbox']['x1']),
                "bbox_h": int(d['bbox']['y2'] - d['bbox']['y1']),
                "gender": g_code
            })
        payload = {
            "boxes": boxes,
            "counts": {
                "camera_id": self.camera_index,
                "date": now_iso,
                "counter": len(detections),
                "male_counter": male_count,
                "female_counter": female_count
            }
        }
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=headers, timeout=5.0)
                if response.status_code not in [200, 201]:
                    logger.warning(f"[API] Push failed with status {response.status_code}")
        except Exception as e:
            logger.debug(f"[API] Error: {e}")

    async def demo_mode(self):
        frame_width, frame_height = 640, 480
        prev_time = datetime.now()
        fps = 0
        demo_count = 0
        logger.info("[DEMO] Demo mode started")
        while self.is_running:
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            for y in range(frame_height):
                intensity = int((y / frame_height) * 50)
                frame[y, :] = [intensity, intensity * 2, intensity // 2]
            num_detections = 2 + (demo_count % 3)
            demo_detections = []
            for i in range(num_detections):
                x_offset = ((demo_count + i * 100) % (frame_width - 100)) 
                y_offset = 100 + i * 120
                demo_detections.append({
                    'x': float(x_offset + 50),
                    'y': float(y_offset + 150),
                    'confidence': 0.85 + (i * 0.05),
                    'gender': 'Male' if i % 2 == 0 else 'Female',
                    'bbox': {
                        'x1': float(x_offset),
                        'y1': float(y_offset),
                        'x2': float(x_offset + 100),
                        'y2': float(y_offset + 150)
                    }
                })
            annotated_frame = self.draw_detections(frame, demo_detections)
            current_time = datetime.now()
            time_diff = (current_time - prev_time).total_seconds()
            if time_diff > 0:
                fps = 1 / time_diff
            prev_time = current_time
            if self.show_fps:
                cv2.putText(annotated_frame, f"FPS: {fps:.1f} (DEMO)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 136), 2)
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            try:
                self.frame_queue.put_nowait((frame_bytes, fps, frame_width, frame_height, demo_detections))
            except asyncio.QueueFull:
                pass
            demo_count += 1
            await asyncio.sleep(0.03)

    async def capture_frames(self):
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.executor, self._init_camera)
            if not self.cap or not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                await self.demo_mode()
                return
            await asyncio.sleep(0.5)
            if self.cap and self.cap.isOpened():
                self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:
                self.frame_width = AppConfig.FRAME_WIDTH
                self.frame_height = AppConfig.FRAME_HEIGHT
            logger.info(f"[CAMERA] Camera {self.camera_index} opened - {self.frame_width}x{self.frame_height}")
            frame_width, frame_height = self.frame_width, self.frame_height
            failed_reads = 0
            max_failed_reads = AppConfig.MAX_FAILED_READS
            frame_count = 0
            target_fps = AppConfig.TARGET_FPS
            frame_interval = 1.0 / target_fps
            fps = 0
            while self.is_running:
                loop_start = datetime.now()
                result = await loop.run_in_executor(self.executor, self._process_one_frame)
                if result is None:
                    failed_reads += 1
                    if failed_reads >= max_failed_reads:
                        if self.cap: self.cap.release()
                        await asyncio.sleep(AppConfig.RETRY_DELAY_SECONDS)
                        await loop.run_in_executor(self.executor, self._init_camera)
                        failed_reads = 0 
                        continue
                    await asyncio.sleep(0.1)
                    continue
                failed_reads = 0
                frame_count += 1
                self.frame_count += 1
                frame_bytes, detections, current_fps = result
                fps = current_fps
                try:
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait((frame_bytes, fps, frame_width, frame_height, detections))
                except Exception:
                    pass
                process_time = (datetime.now() - loop_start).total_seconds()
                sleep_time = max(0, frame_interval - process_time)
                await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            logger.info("[STOP] Capture frames task cancelled")
        except Exception as e:
            logger.error(f"Error in capture_frames: {e}")
        finally:
            if self.cap:
                self.cap.release()
                logger.info("[CAMERA] Camera released")

    def _init_camera(self):
        backend = cv2.CAP_DSHOW if (os.name == 'nt') else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.camera_index, backend)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, AppConfig.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, AppConfig.FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

    def _process_one_frame(self):
        if not self.cap: return None
        current_time = datetime.now()
        time_diff = (current_time - getattr(self, '_last_frame_time', current_time)).total_seconds()
        self._last_frame_time = current_time
        fps = 1.0 / time_diff if time_diff > 0 else 0
        ret, frame = self.cap.read()
        if not ret: return None
        detections = self.detect_persons(frame)
        self.last_detections = detections
        annotated_frame = self.draw_detections(frame, detections)
        if self.show_fps:
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 136), 2)
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes(), detections, fps

    async def broadcast_stats(self):
        logger.info("[STATS] Stats broadcast task started")
        try:
            while self.is_running:
                if self.last_detections is not None:
                    stats = {
                        'type': 'stats',
                        'fps': 30.0,
                        'width': self.frame_width,
                        'height': self.frame_height,
                        'detections': self.last_detections
                    }
                    now = datetime.now()
                    if (now - self.last_api_push).total_seconds() >= 1.0:
                        self.last_api_push = now
                        asyncio.create_task(self._push_to_api(self.last_detections))
                    try:
                        self.stats_queue.put_nowait(stats)
                    except asyncio.QueueFull:
                        try:
                            self.stats_queue.get_nowait()
                            self.stats_queue.put_nowait(stats)
                        except:
                            pass
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info("[STOP] Stats broadcast task cancelled")
        except Exception as e:
            logger.error(f"Error in broadcast_stats: {e}")

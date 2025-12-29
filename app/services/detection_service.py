import torch
import cv2
import numpy as np
import urllib.request
from datetime import datetime
import asyncio
import json
from pathlib import Path
import concurrent.futures
import httpx
import os
from ultralytics import YOLO
from app.config import AppConfig, logger

class DetectionEngine:
    def __init__(self):
        self.model = None
        self.gender_net = None
        self.face_net = None
        self.gender_list = ['Male', 'Female']
        self.track_history = {}
        self.is_running = False
        self.camera_index = AppConfig.CAMERA_INDEX
        self.frame_count = 0
        self.face_check_interval = AppConfig.FACE_CHECK_INTERVAL
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

    def load_model(self, model_path: str = None):
        if model_path is None: model_path = AppConfig.MODEL_PATH
        try:
            self.model = YOLO(model_path)
            gender_model_path = AppConfig.GENDER_MODEL_PATH
            if not Path(gender_model_path).exists():
                logger.warning(f"Custom gender model not found at {gender_model_path}. Please train it first.")
                self.gender_net = None
            else:
                self.gender_net = YOLO(gender_model_path)
            face_display_name = "yolov8n-face.pt"
            face_model_path = AppConfig.FACE_MODEL_PATH
            face_model_url = "https://github.com/lindevs/yolov8-face/releases/download/v1.0.0/yolov8n-face.pt"
            if not Path(face_model_path).exists():
                logger.info(f"Downloading YOLOv8-Face model to {face_model_path}...")
                try:
                    urllib.request.urlretrieve(face_model_url, face_model_path)
                except Exception as e:
                    logger.error(f"Failed to download YOLOv8-Face: {e}")
            self.face_net = YOLO(face_model_path)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(device)
            if self.gender_net:
                self.gender_net.to(device)
            if self.face_net:
                self.face_net.to(device)
            logger.info(f"YOLO detections using device: {device}")
            logger.info("All models loaded successfully (Full CUDA pipeline)")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def _predict_gender(self, frame, person_box, track_id):
        x1, y1, x2, y2 = [int(v) for v in person_box]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1: return "Unknown"
        person_img = frame[y1:y2, x1:x2]
        results = self.face_net(person_img, conf=0.3, verbose=False)
        best_face = None
        max_conf = 0.0
        if results and results[0].boxes:
            for box in results[0].boxes:
                fx1, fy1, fx2, fy2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                if confidence > max_conf:
                    max_conf = confidence
                    best_face = (int(fx1), int(fy1), int(fx2), int(fy2))
        detected_gender = None
        if best_face and self.gender_net:
            fx1, fy1, fx2, fy2 = best_face
            pad_w = int((fx2 - fx1) * 0.1)
            pad_h = int((fy2 - fy1) * 0.1)
            fx1, fx2 = max(0, fx1 - pad_w), min(person_img.shape[1], fx2 + pad_w)
            fy1, fy2 = max(0, fy1 - pad_h), min(person_img.shape[0], fy2 + pad_h)
            if fx2 > fx1 and fy2 > fy1:
                face_img = person_img[fy1:fy2, fx1:fx2]
                results = self.gender_net.predict(face_img, verbose=False)
                if results and results[0].probs:
                    top1_index = results[0].probs.top1
                    detected_gender = results[0].names[top1_index].capitalize()
        return detected_gender

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
                            'male_votes': 0, 
                            'female_votes': 0, 
                            'last_check': 0
                        }
                    hist = self.track_history[track_id]
                    should_check = (self.frame_count - hist['last_check']) > self.face_check_interval
                    if hist['gender'] is None:
                         should_check = True 
                    gender_label = hist['gender'] if hist['gender'] else "Person"
                    if should_check and self.face_net and self.gender_net:
                        try:
                            pred = self._predict_gender(frame, (x1, y1, x2, y2), track_id)
                            hist['last_check'] = self.frame_count
                            if pred:
                                if pred == 'Male': hist['male_votes'] += 1
                                elif pred == 'Female': hist['female_votes'] += 1
                                if hist['male_votes'] > hist['female_votes']:
                                    hist['gender'] = 'Male'
                                elif hist['female_votes'] > hist['male_votes']:
                                    hist['gender'] = 'Female'
                                gender_label = hist['gender']
                        except Exception as e:
                            pass
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
            cv2.rectangle(annotated, (x1, y1), (x2, y2), self.box_color, 2)
            cv2.rectangle(annotated, (x1-1, y1-1), (x2+1, y2+1), (0, 100, 50), 1)
            cv2.circle(annotated, (bottom_x, bottom_y), 8, (0, 212, 255), -1)
            cv2.circle(annotated, (bottom_x, bottom_y), 10, (255, 255, 255), 2)
            label = f"{gender} {conf:.0%}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), self.box_color, -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            if self.show_coords:
                coord_text = f"({bottom_x}, {bottom_y})"
                cv2.putText(annotated, coord_text, (bottom_x + 15, bottom_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 212, 255), 2)
        return annotated

    async def _push_to_api(self, detections):
        if not self.api_url or not self.api_token:
            logger.warning("[API] API URL or token not configured, skipping push")
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
                "camera_id": self.camera_index + 1,
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
                "camera_id": self.camera_index + 1,
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
                if response.status_code == 200 or response.status_code == 201:
                    logger.debug(f"[API] Push successful: {len(detections)} detections, M:{male_count} F:{female_count}")
                else:
                    logger.warning(f"[API] Push failed with status {response.status_code}: {response.text}")
        except httpx.ConnectError as e:
            logger.error(f"[API] Connection error: {e}")
        except httpx.TimeoutException:
            logger.warning("[API] Request timeout after 5s")
        except httpx.HTTPStatusError as e:
            logger.error(f"[API] HTTP error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            logger.error(f"[API] Unexpected error: {type(e).__name__}: {e}")

    async def demo_mode(self):
        frame_width, frame_height = 640, 480
        prev_time = datetime.now()
        fps = 0
        demo_count = 0
        logger.info("[DEMO] Demo mode started - generating synthetic frames")
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
                cv2.putText(annotated_frame, f"FPS: {fps:.1f} (DEMO MODE)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 136), 2)
            cv2.putText(annotated_frame, "NO CAMERA - DEMO MODE", (10, frame_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
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
                logger.info("Running in demo mode with generated frames")
                await self.demo_mode()
                return
            await asyncio.sleep(0.5)
            if self.cap and self.cap.isOpened():
                self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:
                self.frame_width = AppConfig.FRAME_WIDTH
                self.frame_height = AppConfig.FRAME_HEIGHT
            logger.info(f"[CAMERA] Camera {self.camera_index} opened successfully - {self.frame_width}x{self.frame_height}")
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
                    logger.warning(f"Failed to read frame (attempt {failed_reads}/{max_failed_reads})")
                    if failed_reads >= max_failed_reads:
                        logger.error(f"[ERROR] Camera connection lost. Retrying in {AppConfig.RETRY_DELAY_SECONDS}s...")
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
                if frame_count % 60 == 0:
                    logger.info(f"[SUCCESS] Captured {frame_count} frames, queue size: {self.frame_queue.qsize()}, FPS: {fps:.1f}")
                if frame_count % 30 == 0 and detections:
                    logger.info(f"[DETECT] Detected {len(detections)} person(s)")
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
            import traceback
            traceback.print_exc()
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

import torch
import cv2
import numpy as np
import urllib.request
from datetime import datetime
from fastapi import FastAPI, WebSocket, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
from pathlib import Path
import logging
from contextlib import asynccontextmanager

import os
import sys
from logging.handlers import RotatingFileHandler

import concurrent.futures

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO

# --- Production Configuration ---
class AppConfig:
    # Camera
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    TARGET_FPS = 60.0
    
    # AI / Detection
    CONFIDENCE_THRESHOLD = 0.5
    FACE_CONFIDENCE_THRESHOLD = 0.3
    FACE_CHECK_INTERVAL = 4
    
    # Resilience
    MAX_FAILED_READS = 10
    RETRY_DELAY_SECONDS = 5
    
    # Paths
    BASE_DIR = Path(__file__).parent
    LOG_FILE = BASE_DIR / "app.log"
    MODEL_PATH = "yolo11n.pt"
    GENDER_MODEL_PATH = "runs/classify/train/weights/best.pt"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        RotatingFileHandler(AppConfig.LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("VisionTera")

class DetectionEngine:
    def __init__(self):
        self.model = None
        self.gender_net = None
        self.face_net = None
        self.gender_list = ['Male', 'Female']
        self.track_history = {}  # {id: {'gender': None, 'male_votes': 0, 'female_votes': 0, 'last_check': 0}}
        self.is_running = False
        
        # Load defaults from Config
        self.camera_index = AppConfig.CAMERA_INDEX
        self.frame_count = 0
        self.face_check_interval = AppConfig.FACE_CHECK_INTERVAL
        self.confidence = AppConfig.CONFIDENCE_THRESHOLD
        
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
            
            # Load custom trained gender classifier
            gender_model_path = AppConfig.GENDER_MODEL_PATH
            if not Path(gender_model_path).exists():
                logger.warning(f"Custom gender model not found at {gender_model_path}. Please train it first.")
                self.gender_net = None
            else:
                self.gender_net = YOLO(gender_model_path)
            
            # Face Model URLs (ResNet SSD) - Keep face detector
            face_proto = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            face_model = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

            # Paths
            fp_path = Path("face_deploy.prototxt")
            fm_path = Path("face_net.caffemodel")
            
            if not fp_path.exists(): urllib.request.urlretrieve(face_proto, fp_path)
            if not fm_path.exists(): urllib.request.urlretrieve(face_model, fm_path)
                
            self.face_net = cv2.dnn.readNet(str(fp_path), str(fm_path))
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(device)
            if self.gender_net:
                self.gender_net.to(device)

            logger.info(f"YOLO using device: {device}")

            if device == 'cuda':
                try:
                    self.face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    logger.info("OpenCV using CUDA backend")
                except Exception as e:
                    logger.warning(f"Could not set OpenCV to CUDA (using CPU): {e}")

            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def _predict_gender(self, frame, person_box, track_id):
        x1, y1, x2, y2 = [int(v) for v in person_box]
        h, w = frame.shape[:2]
        
        # Ensure box is within frame
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1: return "Unknown"
        
        person_img = frame[y1:y2, x1:x2]
        
        # 1. Detect Face in Person ROI
        blob = cv2.dnn.blobFromImage(person_img, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        faces = self.face_net.forward()
        
        best_face = None
        max_conf = 0.0
        
        # Find best face
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.3:  # Lowered confidence threshold
                fx1 = int(faces[0, 0, i, 3] * person_img.shape[1])
                fy1 = int(faces[0, 0, i, 4] * person_img.shape[0])
                fx2 = int(faces[0, 0, i, 5] * person_img.shape[1])
                fy2 = int(faces[0, 0, i, 6] * person_img.shape[0])
                
                if confidence > max_conf:
                    max_conf = confidence
                    best_face = (fx1, fy1, fx2, fy2)

        # 2. If Face Found -> Predict Gender with YOLO Classifier
        detected_gender = None
        if best_face and self.gender_net:
            fx1, fy1, fx2, fy2 = best_face
            # Add padding for gender model context
            pad_w = int((fx2 - fx1) * 0.1)
            pad_h = int((fy2 - fy1) * 0.1)
            fx1, fx2 = max(0, fx1 - pad_w), min(person_img.shape[1], fx2 + pad_w)
            fy1, fy2 = max(0, fy1 - pad_h), min(person_img.shape[0], fy2 + pad_h)
            
            if fx2 > fx1 and fy2 > fy1:
                face_img = person_img[fy1:fy2, fx1:fx2]
                
                # Use YOLO Classifier
                results = self.gender_net.predict(face_img, verbose=False)
                if results and results[0].probs:
                    top1_index = results[0].probs.top1
                    detected_gender = results[0].names[top1_index].capitalize()
        
        return detected_gender

    def detect_persons(self, frame):
        if self.model is None: return []

        # Tracking enabled
        results = self.model.track(frame, classes=[0], conf=self.confidence, persist=True, verbose=False, tracker="bytetrack.yaml")
        detections = []

        if results and results[0].boxes:
            boxes = results[0].boxes
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                track_id = int(ids[i]) if ids is not None else -1
                
                # History / Tracking Logic
                if track_id != -1:
                    if track_id not in self.track_history:
                        self.track_history[track_id] = {
                            'gender': None, 
                            'male_votes': 0, 
                            'female_votes': 0, 
                            'last_check': 0
                        }
                    
                    hist = self.track_history[track_id]
                    
                    # Check if we should re-run face detection (every N frames OR if gender is unknown)
                    should_check = (self.frame_count - hist['last_check']) > self.face_check_interval
                    if hist['gender'] is None:
                         should_check = True # Always check if unknown, but maybe throttle slightly if desired? 
                         # Actually checking every frame for unknown is good if we want fast initial detection.
                         # But let's throttle slightly to 2 frames to save CPU? No, keep it greedy for start.
                    
                    gender_label = hist['gender'] if hist['gender'] else "Person"

                    if should_check and self.face_net and self.gender_net:
                        try:
                            # Run prediction
                            pred = self._predict_gender(frame, (x1, y1, x2, y2), track_id)
                            hist['last_check'] = self.frame_count
                            
                            if pred:
                                if pred == 'Male': hist['male_votes'] += 1
                                elif pred == 'Female': hist['female_votes'] += 1
                                
                                # Update rigid decision
                                if hist['male_votes'] > hist['female_votes']:
                                    hist['gender'] = 'Male'
                                elif hist['female_votes'] > hist['male_votes']:
                                    hist['gender'] = 'Female'
                                    
                                gender_label = hist['gender']
                        except Exception as e:
                            # logger.error(f"Error predicting gender: {e}")
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

    async def demo_mode(self):
        """Generate demo frames with synthetic detections when camera is not available"""
        frame_width, frame_height = 640, 480
        prev_time = datetime.now()
        fps = 0
        demo_count = 0
        logger.info("[DEMO] Demo mode started - generating synthetic frames")

        while self.is_running:
            # Generate a dark background with some animated elements
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            
            # Add some gradient background
            for y in range(frame_height):
                intensity = int((y / frame_height) * 50)
                frame[y, :] = [intensity, intensity * 2, intensity // 2]

            # Create demo detections (animated boxes)
            num_detections = 2 + (demo_count % 3)
            demo_detections = []

            for i in range(num_detections):
                # Animate boxes across the frame
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

            # Draw annotations
            annotated_frame = self.draw_detections(frame, demo_detections)
            
            # Add FPS
            current_time = datetime.now()
            time_diff = (current_time - prev_time).total_seconds()
            if time_diff > 0:
                fps = 1 / time_diff
            prev_time = current_time

            if self.show_fps:
                cv2.putText(annotated_frame, f"FPS: {fps:.1f} (DEMO MODE)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 136), 2)

            # Add demo indicator
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
            # Run camera initialization in executor
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.executor, self._init_camera)

            if not self.cap or not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                logger.info("Running in demo mode with generated frames")
                # Demo mode: generate test frames instead
                await self.demo_mode()
                return
            
            # Give camera time to warm up
            await asyncio.sleep(0.5)

            # Get properties safely
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
            
            # Target FPS from config
            target_fps = AppConfig.TARGET_FPS
            frame_interval = 1.0 / target_fps
            fps = 0

            while self.is_running:
                loop_start = datetime.now()
                
                # Run the blocking capture and processing in a separate thread
                result = await loop.run_in_executor(self.executor, self._process_one_frame)

                if result is None:
                    failed_reads += 1
                    logger.warning(f"Failed to read frame (attempt {failed_reads}/{max_failed_reads})")
                    
                    if failed_reads >= max_failed_reads:
                        # Production Behavior: Retry indefinitely instead of giving up
                        logger.error(f"[ERROR] Camera connection lost. Retrying in {AppConfig.RETRY_DELAY_SECONDS}s...")
                        if self.cap: self.cap.release()
                        await asyncio.sleep(AppConfig.RETRY_DELAY_SECONDS)
                        
                        # Re-init in thread
                        await loop.run_in_executor(self.executor, self._init_camera)
                        failed_reads = 0 # Reset counter to give new connection a chance
                        continue
                    
                    await asyncio.sleep(0.1)
                    continue
                
                # Reset failed read counter on successful read
                failed_reads = 0
                frame_count += 1
                self.frame_count += 1
                
                # Unpack result
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

                # FPS Control
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
        
        # Calculate FPS inside the thread
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
                        'fps': 30.0,  # Will be updated by client from video
                        'width': self.frame_width,
                        'height': self.frame_height,
                        'detections': self.last_detections
                    }
                    try:
                        self.stats_queue.put_nowait(stats)
                    except asyncio.QueueFull:
                        # Remove old stats and add new one
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

engine = DetectionEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load_model()
    logger.info("Application started")
    yield
    logger.info("Application shutdown")

app = FastAPI(lifespan=lifespan)

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

@app.get("/")
async def get_dashboard():
    return FileResponse(BASE_DIR / "index.html", media_type="text/html")

@app.get("/status")
async def get_status():
    return {
        "status": "ok", 
        "running": engine.is_running,
        "frames_processed": engine.frame_count,
        "camera_index": engine.camera_index
    }

@app.post("/api/start")
async def start_detection(
    camera: int = Query(0),
    confidence: float = Query(0.5),
    show_coords: str = Query("1"),
    show_fps: str = Query("1"),
    box_color: str = Query("00FF88"),
    background_tasks: BackgroundTasks = None
):
    if engine.is_running:
        return {"status": "already_running"}

    engine.camera_index = camera
    engine.confidence = max(0.1, min(1.0, confidence))
    engine.show_coords = show_coords == "1"
    engine.show_fps = show_fps == "1"

    r = int(box_color[4:6], 16)
    g = int(box_color[2:4], 16)
    b = int(box_color[0:2], 16)
    engine.box_color = (b, g, r)

    engine.is_running = True
    
    # Properly create tasks
    try:
        engine.capture_task = asyncio.create_task(engine.capture_frames())
        engine.stats_task = asyncio.create_task(engine.broadcast_stats())
        logger.info(f"Started detection with camera {camera}")
    except Exception as e:
        engine.is_running = False
        logger.error(f"Failed to start tasks: {e}")
        return {"status": "error", "message": str(e)}

    return {"status": "started"}

@app.post("/api/stop")
async def stop_detection():
    engine.is_running = False
    await asyncio.sleep(0.1)
    
    # Cancel tasks if they exist
    if engine.capture_task and not engine.capture_task.done():
        engine.capture_task.cancel()
    if engine.stats_task and not engine.stats_task.done():
        engine.stats_task.cancel()
    
    # Close camera
    if engine.cap:
        engine.cap.release()
        engine.cap = None
    
    await asyncio.sleep(0.2)
    return {"status": "stopped"}

@app.get("/video_feed")
async def video_feed():
    async def frame_generator():
        while engine.is_running:
            try:
                frame_bytes, _, _, _, _ = await asyncio.wait_for(engine.frame_queue.get(), timeout=2.0)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'
                       + frame_bytes + b'\r\n')
            except asyncio.TimeoutError:
                pass

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws/stats")
async def websocket_stats(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            try:
                stats = await asyncio.wait_for(engine.stats_queue.get(), timeout=1.0)
                await websocket.send_text(json.dumps(stats))
            except asyncio.TimeoutError:
                pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

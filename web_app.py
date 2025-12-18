import cv2
import numpy as np
import torch
from datetime import datetime
from fastapi import FastAPI, WebSocket, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
from pathlib import Path
import logging
from contextlib import asynccontextmanager

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectionEngine:
    def __init__(self):
        self.model = None
        self.is_running = False
        self.camera_index = 0
        self.confidence = 0.5
        self.show_coords = True
        self.show_fps = True
        self.box_color = (0, 255, 136)
        self.frame_queue = asyncio.Queue(maxsize=1)
        self.stats_queue = asyncio.Queue(maxsize=1)
        self.cap = None
        self.current_frame = None
        self.last_detections = []
        self.capture_task = None
        self.stats_task = None

    def load_model(self, model_path: str = "yolo11n.pt"):
        try:
            self.model = YOLO(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def detect_persons(self, frame):
        if self.model is None:
            return []

        results = self.model(frame, classes=[0], conf=self.confidence, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())

                    bottom_center_x = (x1 + x2) / 2
                    bottom_center_y = y2

                    detections.append({
                        'x': float(bottom_center_x),
                        'y': float(bottom_center_y),
                        'confidence': conf,
                        'bbox': {
                            'x1': float(x1),
                            'y1': float(y1),
                            'x2': float(x2),
                            'y2': float(y2)
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

            cv2.rectangle(annotated, (x1, y1), (x2, y2), self.box_color, 2)
            cv2.rectangle(annotated, (x1-1, y1-1), (x2+1, y2+1), (0, 100, 50), 1)

            cv2.circle(annotated, (bottom_x, bottom_y), 8, (0, 212, 255), -1)
            cv2.circle(annotated, (bottom_x, bottom_y), 10, (255, 255, 255), 2)

            label = f"Person {conf:.0%}"
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
        logger.info("🎬 Demo mode started - generating synthetic frames")

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
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)  # Use DirectShow on Windows

            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                logger.info("Running in demo mode with generated frames")
                # Demo mode: generate test frames instead
                await self.demo_mode()
                return
            
            # Set camera properties for better compatibility
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Give camera time to warm up
            await asyncio.sleep(0.5)

            prev_time = datetime.now()
            fps = 0
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"📹 Camera {self.camera_index} opened successfully - {frame_width}x{frame_height}")
            
            failed_reads = 0
            max_failed_reads = 10

            frame_count = 0
            while self.is_running:
                ret, frame = self.cap.read()

                if not ret:
                    failed_reads += 1
                    logger.warning(f"Failed to read frame (attempt {failed_reads}/{max_failed_reads})")
                    
                    if failed_reads >= max_failed_reads:
                        logger.error(f"Too many failed reads, switching to demo mode")
                        await self.demo_mode()
                        return
                    
                    await asyncio.sleep(0.1)
                    continue
                
                # Reset failed read counter on successful read
                failed_reads = 0
                frame_count += 1
                
                if frame_count % 30 == 0:
                    logger.info(f"✅ Captured {frame_count} frames, queue size: {self.frame_queue.qsize()}")

                current_time = datetime.now()
                time_diff = (current_time - prev_time).total_seconds()
                if time_diff > 0:
                    fps = 1 / time_diff
                prev_time = current_time

                detections = self.detect_persons(frame)
                self.last_detections = detections
                annotated_frame = self.draw_detections(frame, detections)
                
                if frame_count % 30 == 0 and detections:
                    logger.info(f"🎯 Detected {len(detections)} person(s)")

                if self.show_fps:
                    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 136), 2)

                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()

                try:
                    self.frame_queue.put_nowait((frame_bytes, fps, frame_width, frame_height, detections))
                except asyncio.QueueFull:
                    pass

                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            logger.info("✋ Capture frames task cancelled")
        except Exception as e:
            logger.error(f"Error in capture_frames: {e}")
        finally:
            if self.cap:
                self.cap.release()
                logger.info("📷 Camera released")

    async def broadcast_stats(self):
        logger.info("📊 Stats broadcast task started")
        try:
            while self.is_running:
                if self.last_detections is not None:
                    # Get frame dimensions
                    if self.cap:
                        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        width, height = 640, 480
                    
                    stats = {
                        'type': 'stats',
                        'fps': 30.0,  # Will be updated by client from video
                        'width': width,
                        'height': height,
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
            logger.info("✋ Stats broadcast task cancelled")
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

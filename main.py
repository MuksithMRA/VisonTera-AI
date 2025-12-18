from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import json
import threading
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch

# Fix for PyTorch 2.6+ weights_only default change
# Monkey-patch torch.load to use weights_only=False by default for YOLO model loading
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="YOLO Stream Detection Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
stream_active = False
stream_thread = None
detection_data = []
data_lock = threading.Lock()

# File paths
LIVE_DATA_FILE = Path("live_data.json")
MODEL_PATH = "yolo11n.pt"

class YOLODetector:
    def __init__(self, model_path: str):
        """Initialize YOLO model"""
        try:
            self.model = YOLO(model_path)
            logger.info(f"YOLO model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect_persons(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect persons in frame and return bottom-center coordinates"""
        try:
            results = self.model(frame, classes=[0], conf=0.5)  # Class 0 is person
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        # Calculate bottom-center coordinates
                        bottom_center_x = (x1 + x2) / 2
                        bottom_center_y = y2
                        
                        detections.append({
                            'x': float(bottom_center_x),
                            'y': float(bottom_center_y),
                            'confidence': float(confidence),
                            'timestamp': datetime.now().isoformat(),
                            'bbox': {
                                'x1': float(x1),
                                'y1': float(y1),
                                'x2': float(x2),
                                'y2': float(y2)
                            }
                        })
            
            return detections
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

class VideoStreamProcessor:
    def __init__(self, detector: YOLODetector):
        self.detector = detector
        self.active = False
        self.cap = None
        self.stream_thread = None
        
    def start_stream(self, source: str = 0):
        """Start video stream processing"""
        if self.active:
            logger.warning("Stream already active")
            return
            
        try:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise Exception("Failed to open video source")
                
            self.active = True
            logger.info(f"Video stream started from source: {source}")
            
            # Start processing thread
            self.stream_thread = threading.Thread(target=self._process_frames)
            self.stream_thread.daemon = True
            self.stream_thread.start()
            
            return self.stream_thread
            
        except Exception as e:
            logger.error(f"Failed to start stream: {e}")
            if self.cap:
                self.cap.release()
            raise
    
    def stop_stream(self):
        """Stop video stream processing"""
        self.active = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2.0)
            if self.stream_thread.is_alive():
                logger.warning("Stream thread did not terminate cleanly")
        
        logger.info("Video stream stopped")
    
    def _process_frames(self):
        """Process video frames in separate thread"""
        global detection_data
        
        while self.active:
            try:
                if not self.cap or not self.cap.isOpened():
                    break
                
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame from stream")
                    time.sleep(0.1)
                    continue
                
                if not self.active:
                    break
                
                # Detect persons in frame
                detections = self.detector.detect_persons(frame)
                
                # Update detection data with thread safety
                with data_lock:
                    detection_data = detections
                    self._save_to_json(detections)
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                time.sleep(0.1)
        
        logger.debug("Frame processing thread exited")
    
    def _save_to_json(self, detections: List[Dict[str, Any]]):
        """Save detections to JSON file"""
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'person_count': len(detections),
                'detections': detections
            }
            
            with open(LIVE_DATA_FILE, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save data to JSON: {e}")

# Initialize YOLO detector
try:
    detector = YOLODetector(MODEL_PATH)
    stream_processor = VideoStreamProcessor(detector)
    logger.info("YOLO service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize service: {e}")
    detector = None
    stream_processor = None

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("YOLO Stream Service starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("YOLO Stream Service shutting down...")
    if stream_processor:
        stream_processor.stop_stream()

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "YOLO Stream Detection Service",
        "version": "1.0.0",
        "status": "running",
        "model": "YOLOv11n",
        "endpoints": {
            "stream": "/stream",
            "status": "/status",
            "data": "/data"
        }
    }

@app.get("/status")
async def get_status():
    """Get service status"""
    return {
        "service": "running",
        "model_loaded": detector is not None,
        "stream_active": stream_processor.active if stream_processor else False,
        "live_data_file_exists": LIVE_DATA_FILE.exists()
    }

@app.post("/stream")
async def start_stream(source: int = 0):
    """Start video stream processing"""
    global stream_thread, stream_active
    
    if not detector:
        raise HTTPException(status_code=503, detail="YOLO model not loaded")
    
    if stream_processor.active:
        return {"message": "Stream already active"}
    
    try:
        stream_thread = stream_processor.start_stream(source)
        stream_active = True
        return {"message": "Stream started successfully", "source": source}
    except Exception as e:
        logger.error(f"Failed to start stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/stream")
async def stop_stream():
    """Stop video stream processing"""
    global stream_active
    
    if stream_processor:
        stream_processor.stop_stream()
    
    stream_active = False
    return {"message": "Stream stopped successfully"}

@app.get("/data")
async def get_data():
    """Get current detection data"""
    global detection_data
    
    try:
        if LIVE_DATA_FILE.exists():
            with open(LIVE_DATA_FILE, 'r') as f:
                data = json.load(f)
            return data
        else:
            return {"message": "No data available", "person_count": 0, "detections": []}
    except Exception as e:
        logger.error(f"Failed to read data: {e}")
        raise HTTPException(status_code=500, detail="Failed to read detection data")

@app.get("/test")
async def test_page():
    """Simple test page for the stream endpoint"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLO Stream Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            button { padding: 10px 20px; margin: 10px; font-size: 16px; }
            #status { margin: 20px 0; padding: 10px; background: #f0f0f0; }
            #data { margin: 20px 0; padding: 10px; background: #e8f4f8; }
        </style>
    </head>
    <body>
        <h1>YOLO Stream Detection Test</h1>
        <div id="status">Status: Ready</div>
        <button onclick="startStream()">Start Stream</button>
        <button onclick="stopStream()">Stop Stream</button>
        <button onclick="getData()">Get Data</button>
        <div id="data"></div>
        
        <script>
            async function startStream() {
                try {
                    const response = await fetch('/stream', {method: 'POST'});
                    const result = await response.json();
                    document.getElementById('status').innerHTML = 'Status: ' + result.message;
                } catch (error) {
                    document.getElementById('status').innerHTML = 'Error: ' + error.message;
                }
            }
            
            async function stopStream() {
                try {
                    const response = await fetch('/stream', {method: 'DELETE'});
                    const result = await response.json();
                    document.getElementById('status').innerHTML = 'Status: ' + result.message;
                } catch (error) {
                    document.getElementById('status').innerHTML = 'Error: ' + error.message;
                }
            }
            
            async function getData() {
                try {
                    const response = await fetch('/data');
                    const data = await response.json();
                    document.getElementById('data').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                } catch (error) {
                    document.getElementById('data').innerHTML = 'Error: ' + error.message;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
import os
import sys
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

load_dotenv()

class AppConfig:
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    TARGET_FPS = 60.0
    CONFIDENCE_THRESHOLD = 0.5
    FACE_CONFIDENCE_THRESHOLD = 0.3
    FACE_CHECK_INTERVAL = 4
    MAX_FAILED_READS = 10
    RETRY_DELAY_SECONDS = 5
    BASE_DIR = Path(__file__).parent.parent
    LOG_FILE = BASE_DIR / "app.log"
    MODEL_PATH = "infrastructure/models/yolo11n.pt"
    GENDER_MODEL_PATH = "runs/classify/train/weights/best.pt"
    FACE_MODEL_PATH = "infrastructure/models/yolov8n-face.pt"
    API_URL = os.getenv("base_url_dev", "http://localhost:8080")
    API_TOKEN = os.getenv("access_VT", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        RotatingFileHandler(AppConfig.LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("VisionTera")

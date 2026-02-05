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
    MAX_FAILED_READS = 10
    RETRY_DELAY_SECONDS = 5
    BASE_DIR = Path(__file__).parent.parent
    LOG_FILE = BASE_DIR / "app.log"
    MODEL_PATH = "infrastructure/models/yolo11m.pt"
    GENDER_MODEL_PATH = "infrastructure/models/yolo11m-gender.pt"
    GENDER_CHECK_INTERVAL = 30
    GENDER_VOTING_ENABLED = True
    GENDER_VOTE_THRESHOLD = 3
    API_URL = os.getenv("base_url_dev", "https://visiontera-backend-697124318129.europe-west4.run.app")
    API_TOKEN = os.getenv("access_VT", "eyJhbGciOiJodHRwOi8vd3d3LnczLm9yZy8yMDAxLzA0L3htbGRzaWctbW9yZSNobWFjLXNoYTUxMiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiaWF0IjoiMTIvMjkvMjAyNSAwOTowMTo1NCIsImp0aSI6IjAwNzYzYzA5LTc4ZWItNGU4NC1iOGUyLTY1ODEyNmU1NzJiNSIsInVzZXJfaWQiOiIzMiIsIm1lcmNoYW50X2lkIjoiMCIsImlzX3N0YWZmIjoiRmFsc2UiLCJpc19zdXBlcnVzZXIiOiJGYWxzZSIsImV4cCI6MTc2NzAwMDcxNH0.RfM-73xQ5o2WOZs07zLoB9NvQWDkzSFdDSl9JZefP_owskIsIJNpg0sMv2QVco-hCyuW0OU08A4I7JDYwmIYCw")
    
    RTSP_ENABLED = os.getenv("RTSP_ENABLED", "false").lower() == "true"
    RTSP_URLS = os.getenv("RTSP_URLS", "")
    RTSP_NAMES = os.getenv("RTSP_NAMES", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        RotatingFileHandler(AppConfig.LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("VisionTera")

import asyncio
from app.config import logger
from app.services.state import camera_manager

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from infrastructure.train_yolo_gender import train_yolo_gender
except ImportError:
    pass

async def run_training_task(epochs=50):
    try:
        from infrastructure.train_yolo_gender import train_yolo_gender
        logger.info(f"Starting background training task (YOLO-cls, epochs={epochs})...")
        loop = asyncio.get_running_loop()
        
        best_model_dir = await loop.run_in_executor(None, lambda: train_yolo_gender(epochs=epochs))
        
        if best_model_dir:
            logger.info(f"Training complete. Model saved at {best_model_dir}. Reloading models...")
            camera_manager.load_models()
            logger.info("Models reloaded successfully.")
            return best_model_dir
        else:
            logger.error("Training completed but no model path returned.")
            return None
    except Exception as e:
        logger.error(f"Training task failed: {e}")
        return None

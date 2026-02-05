import asyncio
from app.config import logger
from app.services.state import camera_manager, pipeline

async def run_training_task():
    try:
        logger.info("Starting background training task...")
        loop = asyncio.get_running_loop()
        best_model_path = await loop.run_in_executor(None, pipeline.run_pipeline)
        
        if best_model_path:
            logger.info(f"Training complete. New model at {best_model_path}. Reloading models...")
            camera_manager.load_models()
            logger.info("Models reloaded successfully.")
        else:
            logger.error("Training completed but no model path returned.")
    except Exception as e:
        logger.error(f"Training task failed: {e}")

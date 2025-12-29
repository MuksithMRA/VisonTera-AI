import asyncio
from app.config import logger
from app.services.state import engine, pipeline

async def run_training_task():
    try:
        logger.info("Starting background training task...")
        loop = asyncio.get_running_loop()
        best_model_path = await loop.run_in_executor(None, pipeline.run_pipeline)
        
        if best_model_path:
            logger.info(f"Training complete. New model at {best_model_path}. Reloading engine...")
            engine.load_model()
            logger.info("Engine reloaded with new model.")
        else:
            logger.error("Training completed but no model path returned.")
    except Exception as e:
        logger.error(f"Training task failed: {e}")

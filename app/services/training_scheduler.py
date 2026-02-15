import asyncio
import time
from datetime import datetime
import shutil
import logging
from pathlib import Path
from app.config import AppConfig, logger
from app.services.training_service import run_training_task
from app.infrastructure.inference_engine import InferenceEngine

async def start_pipeline_cycle():
    """
    Orchestrates the full pipeline:
    1. Merge Data (infrastructure/merge_to_dataset.py)
    2. Prepare Data (infrastructure/prepare_yolo_dataset.py)
    3. Train Model (via run_training_task)
    """
    loop = asyncio.get_running_loop()
    
    # 1. Merge Data
    try:
        from infrastructure.merge_to_dataset import merge_labeled_images
        logger.info("Step 1: Merging labeled data...")
        await loop.run_in_executor(None, merge_labeled_images)
    except Exception as e:
        logger.error(f"Merge step failed: {e}")
        return

    # 2. Prepare Data
    try:
        from infrastructure.prepare_yolo_dataset import prepare_dataset
        logger.info("Step 2: Preparing YOLO dataset...")
        stats = await loop.run_in_executor(None, prepare_dataset)
        
        # Check if we have enough data (optional check)
        total_train = stats['train']['FEMALE'] + stats['train']['MALE']
        if total_train < 10:
            logger.warning(f"Not enough training data ({total_train} images). Skipping training.")
            return
            
    except Exception as e:
        logger.error(f"Preparation step failed: {e}")
        return

    # 3. Train Model
    logger.info("Step 3: Starting training...")
    await run_training_task()

async def run_scheduler_loop():
    """Background task for automated training cycle."""
    if not AppConfig.TRAINING_SCHEDULER_ENABLED:
        logger.info("Training scheduler is disabled.")
        return

    # Ensure infrastructure is in path
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))

    logger.info(f"Training scheduler started. Interval: {AppConfig.TRAINING_SCHEDULER_INTERVAL}s")
    last_run_time = time.time()
    
    while True:
        try:
            await asyncio.sleep(60)
            if time.time() - last_run_time >= AppConfig.TRAINING_SCHEDULER_INTERVAL:
                logger.info("Scheduler: Triggering pipeline...")
                await start_pipeline_cycle()
                last_run_time = time.time()
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            await asyncio.sleep(60)

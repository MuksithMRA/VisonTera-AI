import os
import shutil
import json
from datetime import datetime
import logging
from pathlib import Path
from ultralytics import YOLO
import random

from app.config import AppConfig

def split_dataset(source_dir, dest_dir, split_ratio=0.8):
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    if not source_path.exists():
        logging.error(f"Source data directory {source_path} does not exist!")
        return

    classes = [d.name for d in source_path.iterdir() if d.is_dir()]
    logging.info(f"Found classes: {classes}")
    
    for class_name in classes:
        (dest_path / 'train' / class_name).mkdir(parents=True, exist_ok=True)
        (dest_path / 'val' / class_name).mkdir(parents=True, exist_ok=True)
        
        files = list((source_path / class_name).glob('*.*'))
        random.shuffle(files)
        
        split_idx = int(len(files) * split_ratio)
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        logging.info(f"Processing {class_name}: {len(train_files)} training, {len(val_files)} validation")
        
        for f in train_files:
            shutil.copy2(f, dest_path / 'train' / class_name / f.name)
            
        for f in val_files:
            shutil.copy2(f, dest_path / 'val' / class_name / f.name)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] TRIAN_PIPELINE: %(message)s")
logger = logging.getLogger("TrainPipeline")

class TrainingPipeline:
    def __init__(self, data_dir="data", dataset_split_dir="gender_dataset_split", model_output_dir="model_store"):
        self.data_dir = data_dir
        self.dataset_split_dir = dataset_split_dir
        self.output_dir = model_output_dir
        self.is_training = False
        self.status = "idle"
        self.progress = 0

    def run_pipeline(self):
        if self.is_training:
            logger.warning("Training already in progress")
            return False

        try:
            self.is_training = True
            self.status = "preparing_data"
            logger.info("Starting pipeline: Data Preparation")
            
            if os.path.exists(self.dataset_split_dir):
                shutil.rmtree(self.dataset_split_dir)
            
            split_dataset(self.data_dir, self.dataset_split_dir)
            
            self.status = "training"
            logger.info("Starting pipeline: Training")
            
            model_path = AppConfig.BASE_DIR / "infrastructure/models/yolo11n-cls.pt"
            model = YOLO(str(model_path))
            
            import torch
            device = 0 if torch.cuda.is_available() else 'cpu'

            results = model.train(
                data=self.dataset_split_dir, 
                epochs=10, 
                imgsz=224, 
                device=device, 
                project='runs/classify', 
                name='train', 
                exist_ok=True
            )

            print(results)
            
            self.status = "evaluating"
            logger.info("Starting pipeline: Evaluation")
            self.status = "evaluating"
            logger.info("Starting pipeline: Evaluation")
            metrics = model.val()
            
            # Log results
            top1 = metrics.top1
            top5 = metrics.top5
            logger.info(f"Training Completed. Top1 Accuracy: {top1:.4f}, Top5 Accuracy: {top5:.4f}")
            
            # Save history
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "top1": float(top1),
                "top5": float(top5),
                "model_path": str(Path(self.output_dir) / 'train/weights/best.pt')
            }
            
            history_file = Path('training_history.json')
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
            
            history.append(history_entry)
            
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            self.status = "completed"
            logger.info("Pipeline completed successfully")
            
            best_model_path = Path('runs/classify/train/weights/best.pt')
            if best_model_path.exists():
                return str(best_model_path)
            else:
                logger.error("Best model file not found despite training completion")
                return None

        except Exception as e:
            self.status = "error"
            logger.error(f"Pipeline failed: {e}")
            raise e
        finally:
            self.is_training = False

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()

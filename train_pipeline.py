import os
import shutil
import json
from datetime import datetime
import logging
from pathlib import Path
from ultralytics import YOLO
from prepare_data import split_dataset

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
            
            model = YOLO('yolo11n-cls.pt')
            
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

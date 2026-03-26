"""
VisionTera - Uniform Detector Training (Employee Exclusion)
================================================================
Fine-tunes a YOLO detection model to detect uniforms on people.
When a uniform is detected on a person crop, that person is classified
as an employee and excluded from the people count.

Approach:
    - Uses a YOLO DETECTION model (not classification)
    - Trained on uniform detection dataset (Roboflow format)
    - During inference: crop each detected person → run uniform detector
    - If uniform detected on person → employee → exclude from count

Dataset: Roboflow uniform detection dataset (YOLOv11 format)
    datasets/uniform_dataset/
    ├── data.yaml
    ├── train/
    │   ├── images/     # Full images containing people in uniforms
    │   └── labels/     # YOLO format: <class> <x> <y> <w> <h>
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

Usage:
    python research/train_uniform_classifier.py                          # Normal training
    python research/train_uniform_classifier.py --epochs 50              # Custom epochs
    python research/train_uniform_classifier.py --imgsz 640              # Image size
    python research/train_uniform_classifier.py --batch 16               # Batch size
    python research/train_uniform_classifier.py --data path/to/data.yaml # Custom dataset
    python research/train_uniform_classifier.py --freeze 10              # Freeze backbone layers
"""

from ultralytics import YOLO
import os
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - VisionTera.UniformDetector - %(levelname)s - %(message)s',
)
logger = logging.getLogger("VisionTera.UniformDetector")

# ─── Project Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
RESEARCH_DIR = PROJECT_ROOT / "research"
MODELS_DIR = PROJECT_ROOT / "infrastructure" / "models"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets" / "uniform_dataset"

# Base detection model for fine-tuning (smaller model is better for this sub-task)
BASE_MODEL_PATH = MODELS_DIR / "yolo26m.pt"


def get_base_model() -> Path:
    """
    Find the base YOLO detection model for fine-tuning.
    We use a detection model since we're detecting uniform regions on person crops.
    """
    if BASE_MODEL_PATH.exists():
        logger.info(f"✅ Using local detection model: {BASE_MODEL_PATH}")
        return BASE_MODEL_PATH

    # Fallback: auto-download
    logger.warning("⚠️  yolo26m.pt not found locally. YOLO will auto-download it.")
    return Path("yolo11m.pt")


def validate_and_prepare_dataset(dataset_path: Path, data_yaml: str = None) -> str:
    """
    Validate the uniform detection dataset and return path to data.yaml.
    
    Expected Roboflow YOLOv11 detection format:
        uniform_dataset/
        ├── data.yaml
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── valid/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/
    """
    if data_yaml and Path(data_yaml).exists():
        logger.info(f"📄 Using provided data.yaml: {data_yaml}")
        return data_yaml

    # Look for data.yaml in dataset directory
    roboflow_yaml = dataset_path / "data.yaml"
    
    if not roboflow_yaml.exists():
        logger.error(f"""
╔══════════════════════════════════════════════════════════════════╗
║  ⚠️  DATASET NOT FOUND                                          ║
║                                                                  ║
║  Expected: {str(roboflow_yaml):<52s}║
║                                                                  ║
║  Download uniform detection dataset from Roboflow:               ║
║    1. Visit: https://universe.roboflow.com                       ║
║    2. Search for "uniform detection"                             ║
║    3. Export in YOLOv11 format                                   ║
║    4. Extract to: datasets/uniform_dataset/                      ║
╚══════════════════════════════════════════════════════════════════╝
        """)
        raise FileNotFoundError(f"Dataset config not found: {roboflow_yaml}")

    # Read original config
    with open(roboflow_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    logger.info(f"📄 Found data.yaml: {data_config.get('nc', '?')} classes: {data_config.get('names', [])}")

    # Create a training-ready YAML with absolute paths
    fixed_yaml_path = RESEARCH_DIR / "uniform_train.yaml"
    fixed_config = {
        'path': str(dataset_path.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': data_config.get('nc', 1),
        'names': data_config.get('names', ['uniform']),
    }

    with open(fixed_yaml_path, 'w') as f:
        yaml.dump(fixed_config, f, default_flow_style=False)

    logger.info(f"📄 Training config saved: {fixed_yaml_path}")

    # Validate splits
    splits = {
        "Train": dataset_path / "train" / "images",
        "Valid": dataset_path / "valid" / "images",
        "Test":  dataset_path / "test" / "images",
    }

    for split_name, img_dir in splits.items():
        if img_dir.exists():
            img_count = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
            lbl_dir = img_dir.parent / "labels"
            lbl_count = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
            logger.info(f"   📊 {split_name}: {img_count} images, {lbl_count} labels")
        else:
            logger.warning(f"   ⚠️  {split_name} split not found at {img_dir}")

    return str(fixed_yaml_path)


def train_uniform_detector(
    data_yaml: str = None,
    dataset_path: str = None,
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 16,
    freeze: int = 10,
    patience: int = 15,
    resume: bool = False,
):
    """
    Fine-tune YOLO detection model to detect uniforms.
    
    During inference in the pipeline:
        1. Person detected by main model (yolo26m.pt)
        2. Person crop extracted
        3. This uniform detector runs on the crop
        4. If uniform detected → person is employee → excluded from count
    
    Strategy:
    - Transfer learning from yolo26m.pt 
    - Freeze backbone to retain pre-trained features
    - Train detection head to find uniform regions
    - Only 1 class: 'uniform'
    """
    # Resolve dataset path
    dataset_dir = Path(dataset_path) if dataset_path else DEFAULT_DATASET_DIR

    # Validate and prepare dataset
    training_yaml = validate_and_prepare_dataset(dataset_dir, data_yaml)

    # Generate version tag
    version_tag = datetime.now().strftime("uniform_v_%Y%m%d_%H%M%S")

    # Get base model
    base_model = get_base_model()
    logger.info(f"🏗️  Base model: {base_model}")
    logger.info(f"🎯 Task: Uniform detection (employee identification)")
    logger.info(f"📦 Version: {version_tag}")

    # Load the base detection model
    model = YOLO(str(base_model))

    # ─── Training Configuration ───────────────────────────────────────────────
    try:
        results = model.train(
            data=training_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            freeze=freeze,

            # ── Save & Project ──
            project=str(MODELS_DIR),
            name=version_tag,
            exist_ok=False,

            # ── Optimizer ──
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            weight_decay=0.0005,
            warmup_epochs=3,

            # ── Augmentation ──
            mosaic=1.0,
            mixup=0.1,
            scale=0.5,
            degrees=10.0,
            translate=0.1,
            fliplr=0.5,
            flipud=0.0,
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.4,

            # ── Performance ──
            amp=True,
            workers=4,
            seed=42,
            deterministic=True,
            verbose=True,
            plots=True,

            # ── Resume ──
            resume=resume,
        )

        logger.info("=" * 60)
        logger.info("✅ UNIFORM DETECTOR TRAINING COMPLETE!")
        logger.info(f"💾 Model saved to: {results.save_dir}")
        logger.info(f"📊 Best model: {results.save_dir}/weights/best.pt")
        logger.info("=" * 60)

        # Copy best model for easy access
        import shutil
        best_weights = Path(results.save_dir) / "weights" / "best.pt"
        best_copy = Path(results.save_dir) / "best_model.pt"
        if best_weights.exists():
            shutil.copy2(best_weights, best_copy)
            logger.info(f"📋 Copied best weights to: {best_copy}")

        # Deploy instructions
        logger.info("")
        logger.info("🔄 NEXT STEPS:")
        logger.info(f"   1. Test:")
        logger.info(f"      python -c \"from ultralytics import YOLO; m=YOLO('{best_copy}'); m.predict('test.jpg', show=True)\"")
        logger.info(f"   2. Deploy: Copy {best_copy} to infrastructure/models/uniform_classifier.pt")
        logger.info(f"   3. The detection pipeline will automatically detect uniforms on person crops")
        logger.info(f"   4. Persons with uniforms → excluded from count")

        return results

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error(f"""
╔══════════════════════════════════════════════════════════════════╗
║  TROUBLESHOOTING                                                 ║
║                                                                  ║
║  1. Dataset missing?                                             ║
║     → Download uniform detection dataset from Roboflow           ║
║     → Extract to: datasets/uniform_dataset/                      ║
║                                                                  ║
║  2. GPU out of memory?                                           ║
║     → Reduce batch size: --batch 8 (or --batch 4)               ║
║     → Reduce image size: --imgsz 480                             ║
║                                                                  ║
║  3. Labels mismatch?                                             ║
║     → Verify labels exist in train/labels/ and valid/labels/     ║
║     → Format: <class> <x_center> <y_center> <width> <height>    ║
╚══════════════════════════════════════════════════════════════════╝
        """)
        raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="VisionTera - Uniform Detector Training (Employee Exclusion)"
    )
    parser.add_argument('--data', type=str, default=None,
                        help='Path to data.yaml (auto-detected from datasets/uniform_dataset/ if not provided)')
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=80,
                        help='Number of training epochs (default: 80)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for training (default: 640)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--freeze', type=int, default=10,
                        help='Number of backbone layers to freeze (default: 10)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger.info("🚀 VisionTera Uniform Detector - Transfer Learning")
    logger.info(f"   Base: yolo26m.pt (detection model)")
    logger.info(f"   Task: Detect uniforms on person crops")
    logger.info(f"   Purpose: Exclude employees from people count")
    logger.info(f"   Epochs: {args.epochs} | ImgSize: {args.imgsz} | Batch: {args.batch}")
    logger.info(f"   Freeze: {args.freeze} layers | Patience: {args.patience}")

    train_uniform_detector(
        data_yaml=args.data,
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        patience=args.patience,
        resume=args.resume,
    )

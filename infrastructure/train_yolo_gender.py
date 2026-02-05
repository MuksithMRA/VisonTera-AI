"""
YOLO11m-cls Gender Classification Training Script
Optimized for high-angle CCTV surveillance footage.

Usage:
    python infrastructure/train_yolo_gender.py [--epochs N] [--batch N]
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - VisionTera.YOLO - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("VisionTera.YOLO")

def train_yolo_gender(epochs=50, batch_size=64, imgsz=224, patience=10):
    """
    Train YOLO11m-cls for gender classification.
    
    Args:
        epochs: Maximum training epochs
        batch_size: Batch size
        imgsz: Input image size
        patience: Early stopping patience
    """
    from ultralytics import YOLO
    import torch
    
    # Generate version ID
    version_id = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info("="*60)
    logger.info("YOLO11m-cls Gender Classification Training")
    logger.info("="*60)
    logger.info(f"Version: {version_id}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Dataset path
    data_path = Path("datasets/new_dataset_yolo")
    if not data_path.exists():
        logger.error(f"Dataset not found at {data_path}. Run prepare_yolo_dataset.py first!")
        return None
    
    # Count images
    train_female = len(list((data_path / "train" / "FEMALE").glob("*")))
    train_male = len(list((data_path / "train" / "MALE").glob("*")))
    val_female = len(list((data_path / "val" / "FEMALE").glob("*")))
    val_male = len(list((data_path / "val" / "MALE").glob("*")))
    
    logger.info(f"\nDataset: {data_path.absolute()}")
    logger.info(f"Train: {train_female + train_male} (F: {train_female}, M: {train_male})")
    logger.info(f"Val: {val_female + val_male} (F: {val_female}, M: {val_male})")
    
    # Load pretrained YOLO11m-cls
    logger.info("\nLoading pretrained YOLO11m-cls...")
    model = YOLO('yolo11m-cls.pt')
    
    # Training configuration - optimized for surveillance
    logger.info("\nStarting training with surveillance-optimized augmentations...")
    logger.info(f"Epochs: {epochs}, Batch: {batch_size}, Image size: {imgsz}")
    
    results = model.train(
        data=str(data_path.absolute()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=patience,
        
        # Surveillance-optimized augmentations
        degrees=15.0,      # Rotation for camera angle variation
        translate=0.1,     # Slight translation
        scale=0.3,         # Scale variation (people at different distances)
        shear=0.0,         # No shear
        perspective=0.0,   # No perspective transform
        flipud=0.1,        # Vertical flip for top-down views
        fliplr=0.5,        # Horizontal flip
        mosaic=0.0,        # Disabled for classification
        mixup=0.0,         # Disabled for classification
        erasing=0.5,       # Random erasing for occlusion robustness
        
        # Color augmentations for varying lighting
        hsv_h=0.015,       # Hue variation
        hsv_s=0.7,         # Saturation variation
        hsv_v=0.4,         # Brightness variation
        
        # Training settings
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.01,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Regularization
        label_smoothing=0.1,
        dropout=0.5,
        
        # Output configuration
        project='infrastructure/models',
        name=version_id,
        exist_ok=False,
        pretrained=True,
        verbose=True,
        seed=42,
        deterministic=True,
        
        # Performance
        workers=4,
        amp=True,  # Mixed precision
    )
    
    # Training complete
    output_dir = Path("infrastructure/models") / version_id
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    
    # Get metrics
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        logger.info(f"Top-1 Accuracy: {metrics.get('metrics/accuracy_top1', 'N/A'):.4f}")
        logger.info(f"Top-5 Accuracy: {metrics.get('metrics/accuracy_top5', 'N/A'):.4f}")
    
    logger.info(f"\nModel saved to: {output_dir.absolute()}")
    logger.info(f"Best weights: {output_dir / 'weights' / 'best.pt'}")
    
    # Copy best model to standard location for compatibility
    best_pt = output_dir / "weights" / "best.pt"
    if best_pt.exists():
        import shutil
        # Copy as best_model.pt for compatibility with existing versioning
        shutil.copy(best_pt, output_dir / "best_model.pt")
        logger.info(f"Copied to: {output_dir / 'best_model.pt'}")
    
    return str(output_dir)

def main():
    parser = argparse.ArgumentParser(description="Train YOLO11m-cls for gender classification")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    args = parser.parse_args()
    
    result = train_yolo_gender(
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        patience=args.patience
    )
    
    if result:
        print(f"\n✅ Training completed successfully!")
        print(f"   Model saved to: {result}")
    else:
        print("\n❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

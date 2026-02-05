"""
Dataset Preparation Script for YOLO11m-cls Gender Classification
Splits the new_dataset into train/val with YOLO-compatible structure.
"""

import os
import shutil
import random
from pathlib import Path
from datetime import datetime

# Configuration
SOURCE_DIR = Path("datasets/new_dataset")
OUTPUT_DIR = Path("datasets/new_dataset_yolo")
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

def prepare_dataset():
    """Split dataset into train/val folders for YOLO classification."""
    random.seed(RANDOM_SEED)
    
    # Create output directories
    for split in ['train', 'val']:
        for cls in ['FEMALE', 'MALE']:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    
    stats = {'train': {'FEMALE': 0, 'MALE': 0}, 'val': {'FEMALE': 0, 'MALE': 0}}
    
    for cls in ['FEMALE', 'MALE']:
        class_dir = SOURCE_DIR / cls
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found!")
            continue
        
        # Get all images
        images = list(class_dir.glob('*'))
        images = [f for f in images if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy to output directories
        for img in train_images:
            shutil.copy2(img, OUTPUT_DIR / 'train' / cls / img.name)
            stats['train'][cls] += 1
        
        for img in val_images:
            shutil.copy2(img, OUTPUT_DIR / 'val' / cls / img.name)
            stats['val'][cls] += 1
    
    # Print summary
    print("\n" + "="*50)
    print("DATASET PREPARATION COMPLETE")
    print("="*50)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print(f"\nTrain set:")
    print(f"  FEMALE: {stats['train']['FEMALE']:,} images")
    print(f"  MALE:   {stats['train']['MALE']:,} images")
    print(f"  Total:  {stats['train']['FEMALE'] + stats['train']['MALE']:,} images")
    print(f"\nValidation set:")
    print(f"  FEMALE: {stats['val']['FEMALE']:,} images")
    print(f"  MALE:   {stats['val']['MALE']:,} images")
    print(f"  Total:  {stats['val']['FEMALE'] + stats['val']['MALE']:,} images")
    print("\n" + "="*50)
    
    return stats

if __name__ == "__main__":
    print(f"Starting dataset preparation at {datetime.now()}")
    prepare_dataset()

import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, dest_dir, split_ratio=0.8):
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    classes = [d.name for d in source_path.iterdir() if d.is_dir()]
    print(f"Found classes: {classes}")
    
    for class_name in classes:
        # Create destination directories
        (dest_path / 'train' / class_name).mkdir(parents=True, exist_ok=True)
        (dest_path / 'val' / class_name).mkdir(parents=True, exist_ok=True)
        
        # Get all images
        files = list((source_path / class_name).glob('*.*'))
        random.shuffle(files)
        
        split_idx = int(len(files) * split_ratio)
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        print(f"Processing {class_name}: {len(train_files)} training, {len(val_files)} validation")
        
        # Copy files
        for f in train_files:
            shutil.copy2(f, dest_path / 'train' / class_name / f.name)
            
        for f in val_files:
            shutil.copy2(f, dest_path / 'val' / class_name / f.name)

if __name__ == "__main__":
    split_dataset("data", "gender_dataset_split")

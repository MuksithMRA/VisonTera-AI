"""
Merge labeled CCTV images into the main dataset.
Run this after manually sorting images in datasets/to_label/MALE and datasets/to_label/FEMALE.
"""

import shutil
from pathlib import Path

SOURCE_DIR = Path("datasets/to_label")
TARGET_DIR = Path("datasets/new_dataset")

def merge_labeled_images():
    """Merge manually labeled images into the main dataset."""
    
    stats = {'MALE': 0, 'FEMALE': 0}
    
    for gender in ['MALE', 'FEMALE']:
        source_folder = SOURCE_DIR / gender
        target_folder = TARGET_DIR / gender
        
        if not source_folder.exists():
            print(f"Warning: {source_folder} not found")
            continue
        
        target_folder.mkdir(parents=True, exist_ok=True)
        
        images = list(source_folder.glob('*.jpg')) + list(source_folder.glob('*.png'))
        
        for img in images:
            target_path = target_folder / img.name
            
            # Handle duplicates by adding suffix
            if target_path.exists():
                stem = img.stem
                suffix = img.suffix
                counter = 1
                while target_path.exists():
                    target_path = target_folder / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            shutil.copy2(img, target_path)
            stats[gender] += 1
    
    print(f"\n{'='*50}")
    print("MERGE COMPLETE")
    print(f"{'='*50}")
    print(f"Added MALE images: {stats['MALE']}")
    print(f"Added FEMALE images: {stats['FEMALE']}")
    print(f"Total added: {stats['MALE'] + stats['FEMALE']}")
    print(f"\nNow run:")
    print(f"  python infrastructure/prepare_yolo_dataset.py")
    print(f"  python infrastructure/train_yolo_gender.py")
    print(f"{'='*50}")

if __name__ == "__main__":
    merge_labeled_images()

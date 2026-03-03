"""
VisionTera - Dataset Merger
===========================
Combines CrowdHuman (NC=1, person) and SCUT-HEAD (NC=1, head) 
into a single unified dataset for training YOLO26m.

Merged Schema:
- Class 0: person
- Class 1: head
"""

import os
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
SOURCES = [
    {
        'name': 'crowdhuman',
        'path': PROJECT_ROOT / 'research' / 'crowdhuman',
        'class_map': {0: 0},  # Maps source class 0 -> target class 0
        'prefix': 'ch_'
    },
    {
        'name': 'scut_head_a',
        'path': PROJECT_ROOT / 'research' / 'scut-head' / 'PartA',
        'class_map': {0: 1},  # Maps source class 0 -> target class 1
        'prefix': 'sc_a_'
    },
    {
        'name': 'scut_head_b',
        'path': PROJECT_ROOT / 'research' / 'scut-head' / 'PartB',
        'class_map': {0: 1},  # Maps source class 0 -> target class 1
        'prefix': 'sc_b_'
    }
]

TARGET_DIR = PROJECT_ROOT / 'datasets' / 'merged_visiontera'
SPLITS = ['train', 'valid', 'test']

def merge():
    print(f"Starting dataset merge into: {TARGET_DIR}")
    
    # 1. Create target directories
    for split in SPLITS:
        (TARGET_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (TARGET_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

    counts = {s['name']: {'images': 0, 'labels': 0} for s in SOURCES}
    
    # 2. Process each source
    for source in SOURCES:
        src_path = source['path']
        src_name = source['name']
        prefix = source['prefix']
        class_map = source['class_map']
        
        print(f"\nProcessing {src_name}...")
        
        if not src_path.exists():
            print(f"   Warning: Path {src_path} not found. Skipping.")
            continue

        for split in SPLITS:
            # Map standard YOLO structure
            # Note: Roboflow uses 'valid', but some datasets use 'val'
            src_split_names = [split, 'val' if split == 'valid' else split]
            
            img_dir = None
            lbl_dir = None
            
            for s_name in src_split_names:
                potential_img = src_path / s_name / 'images'
                if potential_img.exists():
                    img_dir = potential_img
                    lbl_dir = src_path / s_name / 'labels'
                    break
            
            if not img_dir:
                # Check root if not in subfolders (some Roboflow exports differ)
                continue

            # List and copy images
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG']
            files = []
            for ext in extensions:
                files.extend(list(img_dir.glob(ext)))

            if not files:
                continue

            for img_file in tqdm(files, desc=f"   {split}"):
                new_img_name = f"{prefix}{img_file.name}"
                target_img_path = TARGET_DIR / split / 'images' / new_img_name
                
                # Copy Image
                shutil.copy2(img_file, target_img_path)
                counts[src_name]['images'] += 1
                
                # Copy & Remap Label
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                if lbl_file.exists():
                    with open(lbl_file, 'r') as f:
                        lines = f.readlines()
                    
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if not parts: continue
                        
                        src_id = int(parts[0])
                        if src_id in class_map:
                            new_id = class_map[src_id]
                            new_lines.append(f"{new_id} {' '.join(parts[1:])}\n")
                    
                    target_lbl_path = TARGET_DIR / split / 'labels' / f"{prefix}{img_file.stem}.txt"
                    with open(target_lbl_path, 'w') as f:
                        f.writelines(new_lines)
                    counts[src_name]['labels'] += 1

    # 3. Create data.yaml
    data_yaml = {
        'path': str(TARGET_DIR.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 2,
        'names': ['person', 'head']
    }
    
    with open(TARGET_DIR / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    # 4. Show Stats
    print("\n" + "="*40)
    print("MERGE COMPLETE")
    print("="*40)
    total_imgs = 0
    for src_name, stat in counts.items():
        print(f"{src_name:<15}: {stat['images']} images, {stat['labels']} labels")
        total_imgs += stat['images']
    
    print("-" * 40)
    print(f"Total Combined Images: {total_imgs}")
    print(f"Target Config: {TARGET_DIR / 'data.yaml'}")
    print("="*40)
    print("\nYou can now start training with:")
    print(f"python research/train_head_detector.py --data {TARGET_DIR / 'data.yaml'}")

if __name__ == "__main__":
    merge()

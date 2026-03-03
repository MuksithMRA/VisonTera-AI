"""
VisionTera - Cloud Training on Modal
====================================
Cloud-native training handler for VisionTera AI.
Uses high-performance cloud GPUs (NVIDIA A100 or L4) 
to train our 47k merged dataset at high resolution.

Prerequisites:
1. pip install modal
2. modal auth login
3. modal volume create visiontera-data

Usage:
    modal run research/train_modal.py                    # Train with default GPU (L4)
    modal run research/train_modal.py --high-vram        # Train with A100 (80GB)
"""

import os
from pathlib import Path
import modal

# 1. Image Definition: Install dependencies in the cloud environment
image = (
    modal.Image.debian_slim()
    .pip_install(
        "ultralytics",
        "pyyaml",
        "tqdm"
    )
    .apt_install("libgl1-mesa-glx", "libglib2.0-0") # Required for OpenCV/YOLO
)

# 2. Data Volume: Persistent storage for our 47k images
# You MUST upload your data first using: 
# modal volume put visiontera-data ./datasets/merged_visiontera /
volume = modal.Volume.from_name("visiontera-data")
app = modal.App("visiontera-training")

# ─── Remote Training Function ──────────────────────────────────────────────────
@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="L4", # Default GPU, can be upgraded to A100-40GB or A100-80GB
    timeout=86400, # 24 hour timeout
)
def run_train():
    from ultralytics import YOLO
    import yaml
    
    # Paths inside the cloud container
    dataset_root = Path("/data")
    save_dir = Path("/data/runs")
    save_dir.mkdir(exist_ok=True)
    
    # 1. Update data.yaml for the cloud environment
    # Inside Modal, your images are at /data/train/images etc.
    cloud_config = {
        'path': '/data',
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 2,
        'names': ['person', 'head']
    }
    
    config_path = "/tmp/cloud_data.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(cloud_config, f)
        
    print(f"🚀 Starting High-Accuracy Cloud Training on Modal...")
    
    # 2. Load the medium model architecture (YOLO26m)
    # It will auto-download from Ultralytics inside the container
    model = YOLO("yolo26m.pt")
    
    # 3. Start high-accuracy training (Values optimized for cloud A100/L4)
    model.train(
        data=config_path,
        epochs=100,
        imgsz=1280,        # ULTRA Accuracy resolution (cloud only)
        batch=32,          # High batch size for cloud VRAM
        patience=50,
        freeze=0,          # Unfreeze all layers for the 47k dataset
        
        # Optimizer Configuration
        optimizer='AdamW',
        lr0=0.001,
        
        # Cloud Storage Output
        project='/data/models',
        name='cloud_train_v1',
        
        # Surveillance Augmentations
        mosaic=1.0,
        mixup=0.15,
        scale=0.5,
        copy_paste=0.1,
        label_smoothing=0.1,
        multi_scale=True,
        
        # Performance
        amp=True,
        workers=8,
        plots=True
    )
    
    print("✅ Training complete! Model saved to your Modal Volume at /data/models")

if __name__ == "__main__":
    # Local execution command to trigger the remote function
    modal.runner.deploy_app(app)

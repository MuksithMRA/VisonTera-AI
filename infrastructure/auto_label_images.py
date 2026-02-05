"""
Auto-Label Person Crops using ResNet50 or YOLO-cls Model
Automatically sorts images into MALE/FEMALE folders based on model predictions.
Review and correct any mistakes before merging into the dataset.

Usage:
    python infrastructure/auto_label_images.py [--confidence 0.7]
"""

import os
import sys
import shutil
import torch
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.training_pipeline import ResNet50GenderClassifier

SOURCE_DIR = Path("datasets/to_label/data")
CONFIDENCE_THRESHOLD = 0.8  # Only auto-label if confidence > this

def load_resnet_model():
    """Load the ResNet50 gender model."""
    # Try versioned models first
    models_dir = Path("infrastructure/models")
    version_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("v_")]
    version_dirs.sort(key=lambda x: x.name, reverse=True)
    
    model_path = None
    for v_dir in version_dirs:
        candidate = v_dir / "best_model.pt"
        if candidate.exists():
            # Check if it's a ResNet model (not YOLO)
            checkpoint = torch.load(candidate, map_location='cpu')
            if not (isinstance(checkpoint, dict) and 'model' in checkpoint):
                model_path = candidate
                break
    
    if not model_path:
        default = Path("infrastructure/models/resnet50-gender.pt")
        if default.exists():
            model_path = default
    
    if not model_path:
        print("ERROR: No ResNet50 model found!")
        return None
    
    print(f"Loading ResNet50 from: {model_path}")
    model = ResNet50GenderClassifier(num_classes=2)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Model loaded on: {device}")
    
    return model, device

def predict_gender(model, device, image_path):
    """Predict gender for a single image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None, 0.0
    
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0
    img = (img - np.array([0.485, 0.456, 0.406]).reshape(3,1,1)) / np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
    
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device).float()
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze()
    
    confidence, predicted = torch.max(probs, 0)
    gender = "FEMALE" if predicted.item() == 0 else "MALE"
    
    return gender, confidence.item()

def auto_label_images(confidence_threshold=0.7):
    """Auto-label images using ResNet50 model."""
    
    result = load_resnet_model()
    if result is None:
        return
    model, device = result
    
    # Create output folders
    male_dir = SOURCE_DIR / "MALE"
    female_dir = SOURCE_DIR / "FEMALE"
    uncertain_dir = SOURCE_DIR / "UNCERTAIN"
    
    male_dir.mkdir(exist_ok=True)
    female_dir.mkdir(exist_ok=True)
    uncertain_dir.mkdir(exist_ok=True)
    
    # Get all images in source directory (not in subfolders)
    images = [f for f in SOURCE_DIR.iterdir() 
              if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    if not images:
        print("No images found in datasets/to_label/")
        return
    
    print(f"\nAuto-labeling {len(images)} images...")
    print(f"Confidence threshold: {confidence_threshold}")
    print("-" * 50)
    
    stats = {'MALE': 0, 'FEMALE': 0, 'UNCERTAIN': 0}
    
    for img_path in images:
        gender, confidence = predict_gender(model, device, img_path)
        
        if gender is None:
            print(f"  [ERROR] {img_path.name}")
            continue
        
        if confidence >= confidence_threshold:
            target_dir = male_dir if gender == "MALE" else female_dir
            stats[gender] += 1
        else:
            target_dir = uncertain_dir
            stats['UNCERTAIN'] += 1
        
        shutil.move(str(img_path), str(target_dir / img_path.name))
        print(f"  [{gender}] {img_path.name} ({confidence:.2%})")
    
    print(f"\n{'='*50}")
    print("AUTO-LABELING COMPLETE")
    print(f"{'='*50}")
    print(f"MALE: {stats['MALE']}")
    print(f"FEMALE: {stats['FEMALE']}")
    print(f"UNCERTAIN: {stats['UNCERTAIN']} (requires manual review)")
    print(f"\nNEXT STEPS:")
    print(f"1. Review MALE/ and FEMALE/ folders for mistakes")
    print(f"2. Manually sort UNCERTAIN/ images")
    print(f"3. Run: python infrastructure/merge_to_dataset.py")
    print(f"{'='*50}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Auto-label images using ResNet50")
    parser.add_argument("--confidence", type=float, default=0.7, 
                        help="Confidence threshold (default: 0.7)")
    args = parser.parse_args()
    
    auto_label_images(confidence_threshold=args.confidence)

if __name__ == "__main__":
    main()

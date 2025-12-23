from ultralytics import YOLO
from pathlib import Path
import cv2
import random

def test_model():
    model_path = "runs/classify/train/weights/best.pt"
    if not Path(model_path).exists():
        print("Model file not found!")
        return

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Test on a few random images from the validation set
    val_dir = Path("gender_dataset_split/val")
    classes = ['female', 'male']
    
    print("\nStarting inference test on random validation images...")
    print("-" * 50)
    
    correct = 0
    total = 0
    
    for cls in classes:
        cls_dir = val_dir / cls
        files = list(cls_dir.glob("*.*"))
        # Sample up to 5 images per class
        samples = random.sample(files, min(len(files), 5))
        
        for img_path in samples:
            # Run inference
            results = model(str(img_path), verbose=False)
            
            # Get prediction
            probs = results[0].probs
            top1_idx = probs.top1
            pred_label = results[0].names[top1_idx]
            confidence = probs.top1conf.item()
            
            is_correct = (pred_label == cls)
            if is_correct: correct += 1
            total += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{status} Image: {img_path.name}")
            print(f"   True: {cls} | Pred: {pred_label} ({confidence:.2%})")

    print("-" * 50)
    print(f"Test Accuracy on samples: {correct}/{total} ({correct/total:.1%})")

if __name__ == "__main__":
    test_model()

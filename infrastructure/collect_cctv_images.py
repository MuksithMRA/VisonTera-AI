"""
CCTV Person Crop Collector
Automatically captures person crops from CCTV feed for manual labeling.
Uses YOLO tracking to capture each unique person only once.

Usage:
    python infrastructure/collect_cctv_images.py [--camera 0] [--duration 60] [--interval 2]

After collection, manually sort images into MALE/ and FEMALE/ folders,
then run the dataset preparation and retraining scripts.
"""

import os
import sys
import cv2
import time
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

# Configuration
OUTPUT_DIR = Path("datasets/to_label")
MIN_PERSON_HEIGHT = 100  # Minimum person height in pixels
MIN_PERSON_WIDTH = 50    # Minimum person width in pixels
CONFIDENCE_THRESHOLD = 0.5

def collect_person_crops(camera_index=0, duration_seconds=60, interval_seconds=2):
    """
    Collect person crops from CCTV feed using YOLO tracking.
    Each unique person (by track ID) is captured only once.
    
    Args:
        camera_index: Camera index or RTSP URL
        duration_seconds: How long to collect (in seconds)
        interval_seconds: Seconds between captures
    """
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load YOLO model for person detection
    model_path = "infrastructure/models/yolo26m.pt"
    if not Path(model_path).exists():
        model_path = "yolo26m.pt"
    
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    
    # Open camera
    print(f"Opening camera: {camera_index}")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {camera_index}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {width}x{height}")
    
    start_time = time.time()
    end_time = start_time + duration_seconds
    last_capture = 0
    total_captured = 0
    
    # Track which person IDs have already been captured
    captured_ids = set()
    
    print(f"\n{'='*50}")
    print(f"CCTV Person Crop Collector")
    print(f"{'='*50}")
    print(f"Duration: {duration_seconds} seconds")
    print(f"Capture interval: {interval_seconds} seconds")
    print(f"Duplicate detection: TRACKING (each person captured once)")
    print(f"Output: {OUTPUT_DIR.absolute()}")
    print(f"Press 'q' to stop early")
    print(f"{'='*50}\n")
    
    try:
        while time.time() < end_time:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame, retrying...")
                time.sleep(1)
                continue
            
            current_time = time.time()
            
            # Only process at specified intervals
            if current_time - last_capture >= interval_seconds:
                last_capture = current_time
                
                # Detect and track persons (class 0)
                results = model.track(
                    frame, 
                    classes=[0], 
                    conf=CONFIDENCE_THRESHOLD, 
                    persist=True,
                    verbose=False,
                    tracker="bytetrack.yaml"
                )
                
                for result in results:
                    if result.boxes is None:
                        continue
                    
                    boxes = result.boxes
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                        w, h = x2 - x1, y2 - y1
                        
                        # Filter small detections
                        if h < MIN_PERSON_HEIGHT or w < MIN_PERSON_WIDTH:
                            continue
                        
                        # Get track ID
                        track_id = None
                        if box.id is not None:
                            track_id = int(box.id[0])
                        
                        # Skip if already captured this person
                        if track_id is not None and track_id in captured_ids:
                            continue
                        
                        # Mark as captured
                        if track_id is not None:
                            captured_ids.add(track_id)
                        
                        # Crop person
                        person_crop = frame[y1:y2, x1:x2]
                        
                        # Generate unique filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        id_str = f"_id{track_id}" if track_id else ""
                        filename = f"person_{timestamp}{id_str}.jpg"
                        filepath = OUTPUT_DIR / filename
                        
                        # Save crop
                        cv2.imwrite(str(filepath), person_crop)
                        total_captured += 1
                        
                        print(f"[{total_captured}] Captured: {filename} ({w}x{h})")
            
            # Show preview
            preview = frame.copy()
            cv2.putText(preview, f"Captured: {total_captured} | Tracked IDs: {len(captured_ids)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            remaining = int(end_time - current_time)
            cv2.putText(preview, f"Remaining: {remaining}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("CCTV Collector (Press 'q' to stop)", preview)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopping early...")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print(f"\n{'='*50}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*50}")
    print(f"Unique persons captured: {total_captured}")
    print(f"Total tracked IDs: {len(captured_ids)}")
    print(f"Saved to: {OUTPUT_DIR.absolute()}")
    print(f"\nNEXT STEPS:")
    print(f"1. Run: python infrastructure/auto_label_images.py")
    print(f"2. Review MALE/ and FEMALE/ folders")
    print(f"3. Run: python infrastructure/merge_to_dataset.py")
    print(f"4. Run: python infrastructure/prepare_yolo_dataset.py")
    print(f"5. Run: python infrastructure/train_yolo_gender.py")
    print(f"{'='*50}")

def main():
    parser = argparse.ArgumentParser(description="Collect person crops from CCTV")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--interval", type=float, default=2, help="Seconds between captures")
    
    args = parser.parse_args()
    
    collect_person_crops(
        camera_index=args.camera,
        duration_seconds=args.duration,
        interval_seconds=args.interval
    )

if __name__ == "__main__":
    main()



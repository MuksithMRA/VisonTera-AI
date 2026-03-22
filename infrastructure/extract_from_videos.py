"""
Video Frame Extractor for VisionTera AI
Extracts person crops from video files in a specified directory using YOLO tracking.
"""

import cv2
import time
import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

# Common Logic (similar to collect_cctv_images but for files)
OUTPUT_DIR = Path("datasets/to_label")
MIN_PERSON_HEIGHT = 100
MIN_PERSON_WIDTH = 50
CONFIDENCE_THRESHOLD = 0.5

def process_videos(video_folder, interval_seconds=1.0):
    video_paths = list(Path(video_folder).glob("*.mp4")) + \
                  list(Path(video_folder).glob("*.avi")) + \
                  list(Path(video_folder).glob("*.mov")) + \
                  list(Path(video_folder).glob("*.mkv"))
                  
    if not video_paths:
        print(f"No video files found in {video_folder}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load Model
    model_path = "infrastructure/models/yolo26m.pt"
    if not Path(model_path).exists():
        model_path = "yolo26m.pt"
    
    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    
    total_extracted = 0
    
    for vid_path in video_paths:
        print(f"\nProcessing: {vid_path.name}...")
        cap = cv2.VideoCapture(str(vid_path))
        
        if not cap.isOpened():
            print(f"Could not open {vid_path}")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        
        # Calculate frame skip
        frames_to_skip = int(fps * interval_seconds)
        frame_idx = 0
        captured_ids = set() # Per video tracking
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Skip frames logic
            if frame_idx % frames_to_skip != 0:
                frame_idx += 1
                continue
                
            frame_idx += 1
            
            # Detect
            results = model.track(
                frame, 
                classes=[0], 
                conf=CONFIDENCE_THRESHOLD, 
                persist=True, 
                verbose=False,
                tracker="botsort.yaml"
            )
            
            for result in results:
                if result.boxes is None: continue
                
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    w, h = x2 - x1, y2 - y1
                    
                    if h < MIN_PERSON_HEIGHT or w < MIN_PERSON_WIDTH: continue
                    
                    track_id = int(box.id[0]) if box.id is not None else None
                    
                    if track_id is not None and track_id in captured_ids:
                        continue
                        
                    if track_id is not None:
                        captured_ids.add(track_id)
                        
                    # Crop
                    crop = frame[y1:y2, x1:x2]
                    
                    # Save
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"vid_{vid_path.stem}_{ts}_id{track_id}.jpg"
                    cv2.imwrite(str(OUTPUT_DIR / fname), crop)
                    total_extracted += 1
                    
                    # Simple progress indicator
                    sys.stdout.write(f"\rExtracted: {total_extracted} | Current: {vid_path.name}")
                    sys.stdout.flush()
        
        cap.release()
        
    print(f"\n\nDone! Total images extracted: {total_extracted}")
    print(f"Saved to: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to folder containing videos")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between extractions")
    args = parser.parse_args()
    
    process_videos(args.folder, args.interval)

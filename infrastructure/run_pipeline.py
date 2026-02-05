
import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

def run_step(script_name, description, args=[]):
    """Executes a python script located in the infrastructure folder."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    
    script_path = Path("infrastructure") / script_name
    if not script_path.exists():
        print(f"❌ Error: Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)] + [str(a) for a in args]
    
    try:
        # Use simple printing for subprocess to allow viewing output clearly
        subprocess.check_call(cmd)
        print(f"\n✅ {description} completed successfully.")
        return True
    except subprocess.CalledProcessError:
        print(f"\n❌ {description} failed.")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ {description} interrupted.")
        return False

def main():
    parser = argparse.ArgumentParser(description="VisionTera AI Automation Pipeline")
    parser.add_argument("--mode", choices=['full', 'ingest', 'learning'], default=None, 
                        help="Pipeline mode: 'full' (end-to-end), 'ingest' (collect+label), 'learning' (merge+train)")
    
    # Collection args
    parser.add_argument("--duration", type=str, default="60", help="Collection duration (seconds)")
    parser.add_argument("--camera", type=str, default="0", help="Camera index")
    
    # Training args
    parser.add_argument("--epochs", type=str, default="50", help="Training epochs")
    
    args = parser.parse_args()
    
    # Interactive mode if no arguments provided
    if args.mode is None:
        print("\n" + "="*40)
        print(" 🚀 VISIONTERA AI PIPELINE")
        print("="*40)
        print(" 1. 📷 Ingest Mode    (Cam -> Auto-Label)")
        print(" 2. 🧠 Learning Mode  (Merge -> Train)")
        print(" 3. 🔄 Full Cycle     (Cam -> Label -> Train)")
        print(" 4. 🎬 Video Import   (Folder -> Auto-Label)")
        print("-" * 40)
        
        try:
            choice = input("Select pipeline mode (1-4): ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            return

        if choice == '1': args.mode = 'ingest'
        elif choice == '2': args.mode = 'learning'
        elif choice == '3': args.mode = 'full'
        elif choice == '4': args.mode = 'video'
        else:
            print("Invalid choice. Exiting.")
            return

        # Interactive prompts for Collection settings
        if args.mode in ['ingest', 'full']:
            cam_input = input(f"Enter Camera ID/URL [default: {args.camera}]: ").strip()
            if cam_input: args.camera = cam_input
            
            dur_input = input(f"Enter Duration (seconds) [default: {args.duration}]: ").strip()
            if dur_input: args.duration = dur_input
            
        # Interactive prompt for Video
        if args.mode == 'video':
            vid_folder = input("Enter path to Video Folder: ").strip()
            if not vid_folder:
                print("Folder required.")
                return
            args.video_folder = vid_folder

    # --- VIDEO IMPORT PHASE ---
    if args.mode == 'video':
        # 1. Extract
        if not run_step("extract_from_videos.py", "Extract from Videos", [args.video_folder]): return
        
        # 2. Auto-Label
        if not run_step("auto_label_images.py", "Auto-Labeling", []): return
        
        print("\n✅ Video Import complete. Go to datasets/to_label/ to review.")
        return

    # --- INGEST PHASE ---
    if args.mode in ['full', 'ingest']:
        # 1. Collect
        if not run_step("collect_cctv_images.py", "Data Collection", 
                       ["--duration", args.duration, "--camera", args.camera]): return
        
        # 2. Auto-Label
        if not run_step("auto_label_images.py", "Auto-Labeling", []): return
        
        print("\n" + "*"*60)
        print("🛑 PAUSE FOR MANUAL REVIEW")
        print("*"*60)
        print("1. Go to: datasets/to_label/")
        print("2. Review 'MALE' and 'FEMALE' folders (delete mistakes)")
        print("3. Sort images from 'UNCERTAIN' folder")
        print("*"*60)
        
        if args.mode == 'ingest':
            print("\n✅ Ingest complete. When finished reviewing run:")
            print("   python infrastructure/run_pipeline.py --mode learning")
            return
            
        # Pause for Full Cycle
        try:
            input("\n👉 Press ENTER once manual review is finished to continue to training...")
        except KeyboardInterrupt:
            print("\nPipeline stopped.")
            return

    # --- LEARNING PHASE ---
    if args.mode in ['full', 'learning']:
        # 3. Merge
        if not run_step("merge_to_dataset.py", "Merging to Main Dataset", []): return
        
        # 4. Prepare
        if not run_step("prepare_yolo_dataset.py", "Preparing YOLO Format", []): return
        
        # 5. Train
        if not run_step("train_yolo_gender.py", "Model Training", ["--epochs", args.epochs]): return
        
        print("\n✅ Pipeline Finished Successfully!")

if __name__ == "__main__":
    main()

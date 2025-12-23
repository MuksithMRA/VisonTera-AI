import cv2
import numpy as np
import torch
from ultralytics import YOLO
import sys

# Try initializing model
try:
    model = YOLO("yolo11n.pt")
    print("YOLO initialized.")
except Exception as e:
    print(f"YOLO failed: {e}")


print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available (after imports): {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA failed after imports.")

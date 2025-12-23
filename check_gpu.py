import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("\nCUDA is NOT available. Possible reasons:")
    print("1. NVIDIA drivers are missing or outdated.")
    print("2. Video card is not NVIDIA.")
    print("3. PyTorch was installed without CUDA support (but we installed cu121 version).")

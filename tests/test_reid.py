import numpy as np

# Test with different noise levels  
base = np.random.randn(512).astype(np.float32)
base = base / np.linalg.norm(base)

for noise in [0.01, 0.02, 0.05, 0.1, 0.2]:
    noisy_vec = base + np.random.randn(512).astype(np.float32) * noise
    noisy_vec = noisy_vec / np.linalg.norm(noisy_vec)
    sim = float(np.dot(base, noisy_vec))
    print(f"noise={noise:.2f} => similarity={sim:.4f} {'PASS' if sim > 0.60 else 'FAIL'}")

print("\nBut with REAL Re-ID features from the actual model:")
print("The same person's crops produce embeddings that are clustered,")
print("not random vectors + noise. The real test is with the model.\n")

# Test: what similarity does ImageNet ResNet50 give for the same input?
from app.infrastructure.reid_extractor import ReIDFeatureExtractor
import cv2

ext = ReIDFeatureExtractor(device='cuda', use_half=True)
ext.load()

# Create a fake person crop (will be the same for both)
fake_crop = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)

emb1 = ext.extract(fake_crop)
emb2 = ext.extract(fake_crop)
if emb1 is not None and emb2 is not None:
    sim = float(np.dot(emb1, emb2))
    print(f"Same exact crop, extracted twice: similarity={sim:.4f}")
    
    # Slightly modified crop (brightness change)
    bright_crop = np.clip(fake_crop.astype(np.int16) + 20, 0, 255).astype(np.uint8)
    emb3 = ext.extract(bright_crop)
    if emb3 is not None:
        sim2 = float(np.dot(emb1, emb3))
        print(f"Same crop + brightness change: similarity={sim2:.4f}")

"""
YOLOv11m Head Detection Training Script — SCUT-HEAD Part A Dataset
Fine-tunes the pretrained YOLOv11m detection model for head detection 
in crowded and surveillance scenes.

Usage:
    python scut_head/train_scut_head.py [--epochs 50] [--batch 8] [--imgsz 640]
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - VisionTera.SCUTHead - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("VisionTera.SCUTHead")


# ──────────────────────────────────────────────────────────
# Training function
# ──────────────────────────────────────────────────────────
def train_scut_head(
    epochs: int = 50,
    batch_size: int = 8,
    imgsz: int = 640,
    patience: int = 10,
    pretrained_model: str = None,
):
    """Fine-tune YOLOv11m for head detection on the SCUT-HEAD Part A dataset.

    Args:
        epochs:           Maximum training epochs.
        batch_size:       Batch size.
        imgsz:            Input image size (px, square).
        patience:         Early-stopping patience (epochs without improvement).
        pretrained_model: Path to the pretrained detection model.
                          Defaults to ``yolo11m.pt`` in the repo root.
    """
    from ultralytics import YOLO
    import torch

    # Generate a unique version identifier
    version_id = f"scut_head_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("=" * 60)
    logger.info("YOLOv11m Head Detection Training — SCUT-HEAD Part A")
    logger.info("=" * 60)
    logger.info(f"Version : {version_id}")
    logger.info(f"Device  : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        logger.info(f"GPU     : {torch.cuda.get_device_name(0)}")
        logger.info(
            f"VRAM    : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
            if hasattr(torch.cuda.get_device_properties(0), "total_mem")
            else f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # ── 1. Resolve pretrained model path ──────────────────
    if pretrained_model is None:
        # Look for pretrained model in scut_head/models or repo root
        candidates = [
            Path("scut_head/models/yolo11m.pt"),
            Path("yolo11m.pt"),
        ]
        for c in candidates:
            if c.exists():
                pretrained_model = str(c)
                break

    if pretrained_model is None or not Path(pretrained_model).exists():
        logger.error(
            "Pretrained YOLOv11m model not found. "
            "Please place yolo11m.pt in the repo root or scut_head/models/ "
            "or specify --model path."
        )
        return None

    logger.info(f"Pretrained model: {pretrained_model}")

    # ── 2. Validate dataset ───────────────────────────────
    data_yaml = Path("scut_head/datasets/data.yaml")
    if not data_yaml.exists():
        logger.error(
            f"data.yaml not found at {data_yaml}. "
            "Ensure the SCUT-HEAD Part A dataset is in scut_head/datasets/!"
        )
        return None

    # Quick sanity check on train/val folders
    data_root = data_yaml.parent
    train_imgs = data_root / "train" / "images"
    val_imgs = data_root / "valid" / "images"

    n_train = len(list(train_imgs.glob("*"))) if train_imgs.exists() else 0
    n_val = len(list(val_imgs.glob("*"))) if val_imgs.exists() else 0

    logger.info(f"\nDataset : {data_yaml.absolute()}")
    logger.info(f"Train   : {n_train:,} images")
    logger.info(f"Val     : {n_val:,} images")

    if n_train == 0:
        logger.error("Training set is empty — aborting.")
        return None

    # ── 3. Load model & start training ────────────────────
    logger.info(f"\nLoading pretrained YOLOv11m from: {pretrained_model}")
    model = YOLO(pretrained_model)

    logger.info("\nStarting training …")
    logger.info(f"Epochs: {epochs}  |  Batch: {batch_size}  |  Image size: {imgsz}")

    results = model.train(
        data=str(data_yaml.absolute()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=patience,

        # ── Augmentations (tuned for crowd / surveillance) ──
        degrees=10.0,         # Moderate rotation
        translate=0.1,        # Slight translation
        scale=0.5,            # Scale variation (heads at various distances)
        shear=2.0,            # Slight shear for perspective
        perspective=0.0005,   # Mild perspective warp
        flipud=0.0,           # Vertical flip not useful for heads
        fliplr=0.5,           # Horizontal flip
        mosaic=1.0,           # Mosaic augmentation (great for dense scenes)
        mixup=0.1,            # Light mix-up regularisation
        erasing=0.4,          # Random erasing (occlusion robustness)
        copy_paste=0.1,       # Copy-paste augmentation for small objects

        # ── Colour augmentations ──
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,

        # ── Optimizer ──
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # ── Output ──
        project=str(Path("scut_head/models").resolve()),
        name=version_id,
        exist_ok=False,
        pretrained=True,
        verbose=True,
        seed=42,
        deterministic=True,

        # ── Performance ──
        workers=4,
        amp=True, 
    )

    # ── 4. Post-training summary ──────────────────────────
    output_dir = Path("scut_head/models") / version_id

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        logger.info(
            f"mAP@50    : {metrics.get('metrics/mAP50(B)', 'N/A')}"
        )
        logger.info(
            f"mAP@50-95 : {metrics.get('metrics/mAP50-95(B)', 'N/A')}"
        )
        logger.info(
            f"Precision : {metrics.get('metrics/precision(B)', 'N/A')}"
        )
        logger.info(
            f"Recall    : {metrics.get('metrics/recall(B)', 'N/A')}"
        )

    logger.info(f"\nModel saved to : {output_dir.absolute()}")
    logger.info(f"Best weights   : {output_dir / 'weights' / 'best.pt'}")

    # Copy best weights to a standard location for easy reference
    best_pt = output_dir / "weights" / "best.pt"
    if best_pt.exists():
        import shutil

        dest = output_dir / "best_model.pt"
        shutil.copy(best_pt, dest)
        logger.info(f"Copied to      : {dest}")

        # Also copy to scut_head/models as the canonical head-detection model
        canonical = Path("scut_head/models/yolo11m-head.pt")
        shutil.copy(best_pt, canonical)
        logger.info(f"Canonical copy : {canonical}")

    return str(output_dir)


# ──────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv11m for head detection (SCUT-HEAD Part A)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch", type=int, default=8, help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Input image size (default: 640)"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early-stopping patience (default: 10)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to pretrained YOLOv11m model (default: yolo11m.pt in repo root)",
    )

    args = parser.parse_args()

    result = train_scut_head(
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        pretrained_model=args.model,
    )

    if result:
        print(f"\n Training completed successfully!")
        print(f"   Model saved to: {result}")
        print(f"   Best weights also copied to: scut_head/models/yolo11m-head.pt")
    else:
        print("\n Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

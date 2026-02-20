"""
SCUT-HEAD Dataset Preparation & Validation Script

Validates the pre-formatted SCUT-HEAD Part A dataset and prints a summary
of the dataset statistics.

The dataset is expected to already exist at:
    scut_head/datasets/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

Usage:
    python scut_head/prepare_scut_head.py [--dataset-dir PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────
_DEFAULT_DATASET_DIR = Path("scut_head/datasets")
_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
_SEPARATOR = "=" * 60

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
# Helpers
# ──────────────────────────────────────────────────────────
def _count_boxes(label_dir: Path) -> tuple[list[Path], int]:
    """Return label paths and total bounding-box count in *label_dir*.

    Uses a single pass over the directory and lazily reads each file,
    skipping any that cannot be decoded.
    """
    label_paths: list[Path] = []
    box_count = 0
    for p in label_dir.iterdir():
        if p.suffix.lower() != ".txt":
            continue
        label_paths.append(p)
        try:
            box_count += sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
        except (OSError, UnicodeDecodeError):
            pass
    return label_paths, box_count


def _count_images(image_dir: Path) -> int:
    """Count image files in *image_dir* without materialising a full list."""
    return sum(1 for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)


# ──────────────────────────────────────────────────────────
# Validation logic
# ──────────────────────────────────────────────────────────
def validate_dataset(dataset_dir: Path = _DEFAULT_DATASET_DIR) -> Optional[dict]:
    """Validate the SCUT-HEAD Part A dataset and log statistics.

    Args:
        dataset_dir: Root directory of the SCUT-HEAD dataset.

    Returns:
        A ``dict`` mapping each split name to its statistics,
        or ``None`` if validation fails.
    """
    logger.info(_SEPARATOR)
    logger.info("SCUT-HEAD Part A Dataset Validation")
    logger.info(_SEPARATOR)
    logger.info("Dataset directory: %s", dataset_dir.absolute())

    # ── 1. Check data.yaml exists ─────────────────────────
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        logger.error("data.yaml not found at %s", data_yaml)
        return None

    logger.info(" data.yaml found: %s", data_yaml)

    # ── 2. Validate each split ────────────────────────────
    stats: dict[str, dict[str, int]] = {}
    all_ok = True

    for split in ("train", "valid", "test"):
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"

        if not img_dir.exists():
            logger.error(" Missing: %s", img_dir)
            all_ok = False
            continue

        if not lbl_dir.exists():
            logger.error(" Missing: %s", lbl_dir)
            all_ok = False
            continue

        num_images = _count_images(img_dir)
        label_paths, num_boxes = _count_boxes(lbl_dir)
        num_labels = len(label_paths)

        stats[split] = {
            "images": num_images,
            "labels": num_labels,
            "boxes": num_boxes,
        }

        status = "✅" if num_images > 0 else "⚠️"
        logger.info(
            "%s %5s: %s images, %s labels, %s boxes",
            status, split,
            f"{num_images:,}", f"{num_labels:,}", f"{num_boxes:,}",
        )

        if num_images != num_labels:
            logger.warning(
                "    Mismatch: %d images vs %d labels", num_images, num_labels,
            )

    if not all_ok:
        logger.error("Dataset validation failed!")
        return None

    # ── 3. Summary ────────────────────────────────────────
    grand_images = sum(s["images"] for s in stats.values())
    grand_boxes = sum(s["boxes"] for s in stats.values())

    logger.info("\n%s", _SEPARATOR)
    logger.info("DATASET VALIDATION COMPLETE")
    logger.info(_SEPARATOR)
    logger.info("Total images : %s", f"{grand_images:,}")
    logger.info("Total boxes  : %s", f"{grand_boxes:,}")
    logger.info("Classes      : 1 (head)")
    logger.info("Dataset      : SCUT-HEAD Part A")
    logger.info("\ndata.yaml    : %s", data_yaml.absolute())
    logger.info(_SEPARATOR)

    return stats


# ──────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the SCUT-HEAD Part A dataset.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=_DEFAULT_DATASET_DIR,
        help=f"Root directory of the dataset (default: {_DEFAULT_DATASET_DIR})",
    )
    args = parser.parse_args()

    result = validate_dataset(dataset_dir=args.dataset_dir)

    if result:
        logger.info("\n Dataset is ready!")
        logger.info("   Next step: python scut_head/train_scut_head.py")
    else:
        logger.error("Dataset validation failed!")
        logger.error(
            "   Please ensure the SCUT-HEAD Part A dataset is placed in: %s",
            args.dataset_dir.absolute(),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()


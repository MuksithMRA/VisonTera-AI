# 🧠 SCUT-HEAD Fine-tuning Module

Fine-tune **YOLOv11m** for **head detection** using the [SCUT-HEAD Part A dataset](https://universe.roboflow.com/viet-hoang-zbzwq/scut-head-part-a/dataset/1).

---

## 📋 Overview

| Property | Value |
|----------|-------|
| Task | **Object Detection** (bounding boxes around heads) |
| Base Model | `yolo11m.pt` (YOLOv11m detection) |
| Dataset | SCUT-HEAD Part A (Roboflow, YOLO format) |
| Classes | 1 — `head` |
| Output Model | `scut_head/models/yolo11m-head.pt` |

---

## 📁 Dataset Structure

The SCUT-HEAD Part A dataset (pre-formatted in YOLO format from Roboflow) is located at `scut_head/datasets/`:

```
scut_head/datasets/
├── data.yaml
├── train/
│   ├── images/       (1,100 images)
│   └── labels/
├── valid/
│   ├── images/       (400 images)
│   └── labels/
└── test/
    ├── images/       (500 images)
    └── labels/
```

Total: **2,000 images** with head bounding-box annotations.

---

## 🚀 Quick Start

### Step 1: Validate the dataset
```bash
python scut_head/prepare_scut_head.py
```

This checks that the dataset structure is correct and prints statistics.

### Step 2: Train the model
```bash
python scut_head/train_scut_head.py
```

Options:
- `--epochs 50` — Number of training epochs
- `--batch 8` — Batch size (reduce if GPU OOM)
- `--imgsz 640` — Input image size
- `--patience 10` — Early-stopping patience
- `--model PATH` — Custom pretrained model path

---

## 📂 Output Structure

After training, results are saved to:
```
scut_head/models/
├── scut_head_YYYYMMDD_HHMMSS/   # Versioned training run
│   ├── weights/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── best_model.pt            # Copy of best.pt
│   ├── results.csv
│   └── ...
└── yolo11m-head.pt              # Canonical latest model
```

---

## 🔗 Integration

The trained `yolo11m-head.pt` model can be loaded in the detection engine 
to add head-detection capabilities alongside the existing person detection 
and gender classification pipeline.

---

## 📊 SCUT-HEAD Part A Dataset Details

| Property | Value |
|----------|-------|
| Total Images | 2,000 |
| Train | 1,100 |
| Validation | 400 |
| Test | 500 |
| Source | Classroom surveillance |
| Annotation Format | YOLO (via Roboflow) |

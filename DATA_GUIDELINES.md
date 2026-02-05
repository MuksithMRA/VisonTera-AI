# Data Guidelines for VisionTera AI (Gender Classification)

This document outlines the end-to-end workflow for collecting, labeling, and training the Gender Classification model.

## 1. Project Objective
The goal is to train a YOLOv11 Classification model to accurately distinguish between **MALE** and **FEMALE** subjects in CCTV/surveillance footage.

## 2. Automated Data Workflow

We have a master pipeline script to automate the entire process.

### Master Command
Run the interactive pipeline:
```bash
python infrastructure/run_pipeline.py
```
*You will be prompted to choose a mode: Ingest, Learning, or Full Cycle.*

### Phase 1: Ingest (Mode 1)
1. **Data Collection**: Automatically captures crops of people.
   - Captures for 60s by default (configurable).
   - Saves to `datasets/to_label/`.
2. **Auto-Labeling**: Sorts images into `MALE`, `FEMALE`, and `UNCERTAIN`.

### Phase 2: Manual Review (Critical)
**The pipeline pauses here.** You must manually:
1. Go to `datasets/to_label/MALE` and **delete/move** any wrong images.
2. Go to `datasets/to_label/FEMALE` and **delete/move** any wrong images.
3. Check `datasets/to_label/UNCERTAIN` and manually move images to the correct folder.
4. **Delete** any bad quality / blurry images.

### Phase 3: Learning (Mode 2)
Once review is done, this mode runs the rest:
1. **Merge**: Moves clean data from `to_label/` to the master `new_dataset/`.
2. **Prepare**: Splits data into Train (80%) and Val (20%) for YOLO.
3. **Train**: Trains the model (`epochs=50` default).

## 3. Directory Structure

- **`datasets/to_label/`**: Temporary staging area for new captures.
- **`datasets/new_dataset/`**: The MASTER collection of all your source images.
- **`datasets/new_dataset_yolo/`**: The generated, split dataset for YOLO (do not edit manually).

## 4. Image Requirements

### 4.1. Format
- **File Type**: `.jpg` (preferred) or `.png`.
- **Naming**: Handled by automation scripts.

### 4.2. Content
- **Single Subject**: Each image must contain **only one** person. (Handled by collector)
- **Cropping**: Images should be **cropped** to the person's bounding box.
- **Resolution**: Width > 64px recommended.

### 4.3. Quality
- **Camera Angles**: Priority on **CCTV/Surveillance angles**.
- **Diversity**: Varied lighting, clothing, and poses.

## 5. Labeling Criteria

| Class | Criteria |
| :--- | :--- |
| **MALE** | Subject is identifiable as male based on clothing, appearance, or context. |
| **FEMALE** | Subject is identifiable as female. Note: Includes subjects wearing Abaya/Hijab. |

> **Rule of Thumb**: If you can't tell the gender in 2 seconds, delete the image. Don't let the model guess on bad data.

# Data Guidelines for VisionTera AI (Gender Classification)

This document outlines the requirements and format for data collection/annotation for the Gender Classification model.

## 1. Project Objective
The goal is to train a YOLOv11 Classification model to accurately distinguish between **MALE** and **FEMALE** subjects in CCTV/surveillance footage.

## 2. Directory Structure (Deliverable Format)
The data must be organized into a folder-based structure where the folder name represents the class label.

```text
dataset_root/
├── MALE/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   ├── ...
│   └── image_N.jpg
│
└── FEMALE/
    ├── image_001.jpg
    ├── image_002.jpg
    ├── ...
    └── image_N.jpg
```

## 3. Image Requirements

### 3.1. Format
- **File Type**: `.jpg` (preferred) or `.png`.
- **Naming Convention**: 
  - Use unique, alphanumeric filenames (e.g., `cam01_seq05_frame100.jpg` or `uuid.jpg`).
  - Avoid special characters or spaces in filenames.

### 3.2. Content & Cropping
- **Single Subject**: Each image must contain **only one** clearly visible person.
- **Cropping**: Images should be **cropped** to the person's bounding box.
  - Include the full body if possible.
  - Minimum aspect ratio should generally be vertical (height > width).
  - **Do not** provide full wide-angle CCTV frames; we need individual person crops.
- **Resolution**:
  - Minimum recommended: **64 x 128 px**.
  - Higher resolution is better, but consistency is key.

### 3.3. Quality & Diversity
- **Camera Angles**: Priority is on **CCTV/Surveillance angles** (overhead, high-angle).
- **Diversity**: Include varied lighting (day/night/indoor/outdoor), clothing types, and poses.
- **Blur/Occlusion**:
  - Slight blur is acceptable (as real CCTV is often blurry).
  - Severely occluded or unrecognizable subjects should be discarded.

## 4. Labeling Criteria

| Class | Criteria |
| :--- | :--- |
| **MALE** | Subject is identifiable as male based on clothing, appearance, or context. |
| **FEMALE** | Subject is identifiable as female. Note: Includes subjects wearing Abaya/Hijab if applicable to the region. |

> **Note**: If a subject's gender cannot be determined with reasonable confidence (e.g., low resolution, back turned, unisex clothing), **exclude** the image. Do not guess.

## 5. Quality Control Checklist
Before delivery, please ensure:
- [ ] No duplicate images.
- [ ] No corrupted files (0 byte files or unreadable headers).
- [ ] All images are sorted into the correct `MALE` or `FEMALE` folder.
- [ ] No empty folders.

# 🎯 Cross-Camera Person Re-Identification (Re-ID)

> **Module**: VisionTera AI — Multi-Camera Detection Analytics
> **Version**: 1.0
> **Date**: February 24, 2026
> **Status**: ✅ Production Ready

---

## 📋 Overview

Cross-Camera Re-Identification (Re-ID) is an AI-powered feature that enables **accurate person counting across multiple camera feeds** by recognizing the same individual appearing in different cameras. Without Re-ID, a person visible on Camera A and Camera B would be counted as **two separate people**. With Re-ID, they are recognized as the **same person** and counted only **once**.

---

## 🧠 How It Works

### The Problem

In a multi-camera surveillance setup (e.g., a retail store with 4 cameras), a single customer walking through the space may appear on multiple cameras simultaneously or sequentially. A naive system would count this customer once per camera — inflating the total count.

### The Solution

Re-ID uses a **deep learning model** to extract a unique visual "fingerprint" (embedding) for each detected person. These fingerprints are compared across cameras:

- ✅ **Same person on Camera A & B** → Matched → Counted once
- ✅ **Different people on Camera A & B** → Not matched → Counted separately
- ✅ **Person leaves and re-enters** → Re-matched by appearance → Same ID retained

---

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Camera 1   │    │  Camera 2   │    │  Camera N   │
│  (RTSP/USB) │    │  (RTSP/USB) │    │  (RTSP/USB) │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────┐
│              Person Detection (YOLOv11)             │
│         Detects & tracks people in each feed        │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│          Re-ID Feature Extraction (ResNet50)        │
│    Extracts 512-dimensional visual fingerprint      │
│    for each detected person                         │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│        Cross-Camera Re-ID Manager                   │
│                                                     │
│  ┌──────────────────────────────────────────┐       │
│  │         Global Person Gallery            │       │
│  │                                          │       │
│  │  Person 1: 🔵 Cam A + Cam B  (Male)     │       │
│  │  Person 2: 🟢 Cam A only     (Female)   │       │
│  │  Person 3: 🟡 Cam B + Cam C  (Male)     │       │
│  └──────────────────────────────────────────┘       │
│                                                     │
│  • Similarity matching (cosine similarity)          │
│  • Duplicate merging                                │
│  • Stale entry cleanup                              │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              Dashboard & API                        │
│         Deduplicated counts in real-time            │
└─────────────────────────────────────────────────────┘
```

---

## 🔄 Processing Pipeline

### Step 1 — Person Detection

Each camera feed is processed by a **YOLOv11** object detection model trained specifically for person/crowd detection. The model outputs:

- Bounding boxes around each person
- Confidence scores
- Per-camera tracking IDs (via **ByteTrack**)

### Step 2 — Feature Extraction

For each detected person, a **ResNet50-based feature extractor** generates a **512-dimensional embedding vector** — a compact numerical representation of the person's visual appearance.

| Property | Value |
| --- | --- |
| Model Backbone | ResNet50 (ImageNet pre-trained) |
| Embedding Size | 512 dimensions |
| Normalization | L2-normalized |
| Precision | FP16 (GPU-accelerated) |
| Extraction Rate | Every 60 frames per track (~4 seconds) |

### Step 3 — Cross-Camera Matching

The Re-ID Manager maintains a **global person gallery** and performs matching:

1. **New detection arrives** from Camera X with track ID and embedding
2. **Search the gallery** for existing persons with similar embeddings
3. **Cosine similarity** is computed between the new embedding and each gallery member
4. If similarity > **0.60 threshold** → **Match found** → Same person → Assign existing global ID
5. If no match → **New person** → Create new global ID and add to gallery

### Step 4 — Deduplication & Merge

A periodic **merge pass** runs to reconcile any temporary duplicates caused by simultaneous detection across cameras:

- Compares all gallery members from **different** cameras
- Merges persons with high similarity (> 0.60) into a single global ID
- Removes orphaned entries when tracking IDs change

### Step 5 — Real-Time Dashboard

The frontend dashboard displays **deduplicated counts** sourced directly from the Re-ID Manager's authoritative gallery:

| Metric | Description |
| --- | --- |
| Total Detected | Unique individuals across all cameras |
| Male | Unique males detected |
| Female | Unique females detected |
| Peak Count | Maximum simultaneous unique persons |

---

## 📊 Key Features

### ✅ Accurate Cross-Camera Deduplication

The same person appearing on multiple cameras is counted only once. The system handles:

- **Simultaneous appearance** on 2+ cameras
- **Sequential appearance** (person walks from Camera A's view to Camera B's view)
- **Re-appearance** after temporary absence (up to 5 minutes)

### ✅ Gender-Aware Counting

Each unique person is classified as Male or Female using a dedicated **YOLO-cls gender classification model** with a voting mechanism for stability:

- Minimum 3 votes before gender assignment
- Confidence threshold: 60%
- Gender persists across camera transitions

### ✅ Track ID Transition Handling

When the per-camera tracker (ByteTrack) loses and re-acquires a person (assigning a new track ID), the Re-ID system:

1. Detects that the old track is no longer active
2. Matches the new track to the existing global person via appearance similarity
3. Cleans up old track mappings automatically

### ✅ Real-Time Performance

| Metric | Value |
| --- | --- |
| Processing Speed | 15+ FPS per camera |
| Re-ID Extraction | ~30ms per person |
| Match Latency | < 1ms per comparison |
| Gallery Capacity | Up to 100+ simultaneous persons |
| GPU Utilization | Shared with detection model |

### ✅ Automatic Maintenance

- **Stale cleanup**: Persons not seen for 5 minutes are automatically removed from the gallery
- **Duplicate merge**: Periodic reconciliation pass merges duplicates every ~2 seconds
- **Orphan removal**: When track IDs change, orphaned gallery entries are cleaned up

---

## 🔧 Configuration

| Parameter | Default | Description |
| --- | --- | --- |
| `similarity_threshold` | 0.60 | Minimum cosine similarity for a match |
| `max_gallery_size` | 10 | Embeddings retained per person (FIFO) |
| `stale_timeout` | 300s | Time before an unseen person is removed |
| `reid_refresh_interval` | 60 frames | Frames between Re-ID extractions per track |
| `min_crop_height` | 50px | Minimum crop size for extraction |
| `min_crop_width` | 30px | Minimum crop size for extraction |

---

## 📡 API Endpoints

### Get Deduplicated Counts

```
GET /api/reid/counts
```

**Response:**

```json
{
    "total": 5,
    "male": 3,
    "female": 2,
    "unknown": 0
}
```

### Get Currently Visible Persons

```
GET /api/reid/visible
```

**Response:**

```json
{
    "total": 3,
    "male": 2,
    "female": 1,
    "unknown": 0,
    "global_ids": [1, 4, 7]
}
```

### Get System Statistics

```
GET /api/reid/stats
```

**Response:**

```json
{
    "total_global_persons": 12,
    "total_track_mappings": 8,
    "total_cross_camera_matches": 45,
    "total_new_persons": 12,
    "similarity_threshold": 0.6,
    "next_global_id": 13
}
```

---

## 🧪 Testing & Validation

### Test Scenarios Verified

| Scenario | Expected | Result |
| --- | --- | --- |
| 1 person, 2 cameras | Total = 1 | ✅ Pass |
| 2 people, 2 cameras | Total = 2 | ✅ Pass |
| 3 people, 2 cameras | Total = 3 | ✅ Pass |
| Person leaves & re-enters | Same global ID | ✅ Pass |
| Track ID changes (ByteTrack) | Same global ID | ✅ Pass |
| Camera added after detection starts | Matches existing persons | ✅ Pass |
| Camera removed | Persons cleaned up | ✅ Pass |

### Similarity Score Ranges (Observed)

| Comparison Type | Similarity Range |
| --- | --- |
| Same person, same camera | 0.95 – 0.99 |
| Same person, different camera | 0.67 – 0.93 |
| Different persons | 0.10 – 0.45 |

---

## 🛡️ Edge Cases Handled

| Edge Case | How It's Handled |
| --- | --- |
| Two people on same camera | Same-camera skip rule prevents false merges |
| Person partially occluded | Minimum crop size filter skips unreliable crops |
| Rapid track ID churn | Gallery deduplication + orphan cleanup |
| Camera disconnects | `clear_camera()` removes all tracks for that camera |
| Simultaneous first detection | Periodic merge pass reconciles duplicates within ~2s |
| Half-precision (FP16) artifacts | Embeddings cast to FP32 before normalization |

---

## 🚀 Future Enhancements

| Enhancement | Description | Priority |
| --- | --- | --- |
| Custom Re-ID weights | Train on site-specific appearance data | Medium |
| Trajectory analysis | Track person paths across camera zones | High |
| Dwell time analytics | Measure time spent in each zone | High |
| Heatmap generation | Visualize traffic density over time | Medium |
| Historical Re-ID | Match against a database of known individuals | Low |

---

## 📦 Technical Stack

| Component | Technology |
| --- | --- |
| Person Detection | YOLOv11 (custom crowd-trained) |
| Person Tracking | ByteTrack |
| Re-ID Feature Extraction | ResNet50 (ImageNet backbone) |
| Gender Classification | YOLO-cls (custom trained) |
| Backend Framework | FastAPI + Uvicorn |
| Real-Time Communication | WebSocket |
| GPU Acceleration | CUDA (FP16 inference) |
| Frontend | Vanilla JS + WebSocket |

---

> 💡 **Summary**: The Cross-Camera Re-ID system transforms VisionTera from a per-camera detection tool into a **unified multi-camera analytics platform**. By assigning each individual a unique global identity, the system provides accurate, deduplicated person counts — essential for crowd management, retail analytics, and security applications.

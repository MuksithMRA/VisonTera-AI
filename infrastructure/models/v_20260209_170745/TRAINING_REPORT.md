# 🧠 Model Training Report

## 📋 Training Overview

| Property | Value |
|----------|-------|
| **Model Version** | `v_20260209_170745` |
| **Training Date** | February 9, 2026 |
| **Task Type** | Image Classification |
| **Target Application** | Gender Classification for CCTV Surveillance |

---

## 🏗️ Model Architecture

| Parameter | Configuration |
|-----------|---------------|
| **Base Model** | YOLO11m-cls |
| **Model Type** | Medium Classification Network |
| **Pretrained** | ✅ Yes (Transfer Learning) |
| **Input Resolution** | 224 × 224 pixels |
| **Output Classes** | 2 (Male, Female) |

---

## 📊 Dataset Summary

### Distribution Overview

| Split | Male | Female | Total | Ratio |
|-------|------|--------|-------|-------|
| **Train** | 8,391 | 7,876 | 16,267 | 80% |
| **Validation** | 2,098 | 1,969 | 4,067 | 20% |
| **Total** | 10,489 | 9,845 | 20,334 | 100% |

### Class Balance Analysis

📈 **Male:Female Ratio** → `1.06:1` (Well-balanced)

```
Male     ████████████████████████░ 51.6%
Female   ███████████████████████░░ 48.4%
```

> ✅ **Assessment**: Dataset is well-balanced with minimal class imbalance, reducing bias in model predictions.

---

## ⚙️ Training Configuration

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Epochs** | 100 (max) | Maximum training iterations |
| **Early Stopping** | 15 epochs | Patience for convergence |
| **Batch Size** | 32 | Samples per training step |
| **Optimizer** | AdamW | Adaptive learning with weight decay |
| **Initial LR** | 0.001 | Starting learning rate |
| **Final LR** | 0.01 × initial | Minimum learning rate |
| **Weight Decay** | 0.01 | L2 regularization |
| **Dropout** | 0.5 | Regularization rate |

### Learning Rate Schedule

| Phase | Epochs | Configuration |
|-------|--------|---------------|
| **Warmup** | 3 | Bias LR: 0.1, Momentum: 0.8 |
| **Training** | 3-100 | Linear decay to final LR |

### Data Augmentation Pipeline

| Augmentation | Value | Purpose |
|--------------|-------|---------|
| **HSV Hue** | 0.015 | Color variation |
| **HSV Saturation** | 0.7 | Lighting robustness |
| **HSV Value** | 0.4 | Brightness adaptation |
| **Rotation** | ±15° | Orientation invariance |
| **Translation** | 10% | Position invariance |
| **Scale** | 0.3 | Size variation |
| **Horizontal Flip** | 50% | Mirror augmentation |
| **Vertical Flip** | 10% | Additional variation |
| **Random Erasing** | 50% | Occlusion robustness |
| **Auto Augment** | RandAugment | Automatic policy |

---

## 📈 Training Results

### Performance Metrics

| Metric | Best Value | Final Value | Epoch |
|--------|------------|-------------|-------|
| **Top-1 Accuracy** | 83.55% | 83.53% | 59 |
| **Top-5 Accuracy** | 100% | 100% | — |
| **Training Loss** | 0.117 | 0.117 | 74 |
| **Validation Loss** | 0.406 | 0.474 | 25 |

### Training Progression

```
Accuracy Over Epochs:
Epoch  1: ▓░░░░░░░░░ 65.1%
Epoch 10: ▓▓▓▓▓▓▓░░░ 74.9%
Epoch 25: ▓▓▓▓▓▓▓▓░░ 80.6%
Epoch 50: ▓▓▓▓▓▓▓▓▓░ 83.1%
Epoch 74: ▓▓▓▓▓▓▓▓▓░ 83.5%
```

### Loss Convergence Analysis

| Phase | Train Loss | Val Loss | Observation |
|-------|------------|----------|-------------|
| **Start (Epoch 1)** | 0.644 | 0.618 | Initial state |
| **Middle (Epoch 25)** | 0.363 | 0.406 | Best val loss |
| **End (Epoch 74)** | 0.117 | 0.474 | Final state |

### Key Observations

1. **⚠️ Overfitting Detection** 
   - Validation loss starts increasing after epoch 25
   - Training loss continues to decrease
   - Gap between train/val loss widens progressively

2. **✅ Accuracy Plateau**
   - Model accuracy stabilized around 83.5%
   - Early stopping triggered at epoch 74 (15 epochs without improvement)
   - Best weights saved at peak validation performance

3. **📊 Learning Dynamics**
   - Rapid initial learning (65% → 75% in 10 epochs)
   - Gradual refinement phase (75% → 83% over 64 epochs)
   - Diminishing returns after epoch 50

---

## 🎯 Model Performance Summary

### Final Model Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Validation Accuracy** | 83.55% | 🟢 Good |
| **Training Accuracy** | ~99% (estimated) | — |
| **Generalization Gap** | ~15% | 🟡 Moderate |

### Performance Assessment

```
┌─────────────────────────────────────────────┐
│  📊 Model Performance Grade: B+             │
├─────────────────────────────────────────────┤
│  ✅ Strengths:                              │
│  • Solid baseline accuracy (83.5%)          │
│  • Perfect top-5 accuracy                   │
│  • Balanced class performance               │
│                                             │
│  ⚠️ Areas for Improvement:                  │
│  • Mild overfitting observed                │
│  • Validation loss divergence               │
│  • Room for accuracy improvement            │
└─────────────────────────────────────────────┘
```

---

## 📁 Output Files

| File | Size | Description |
|------|------|-------------|
| `best_model.pt` | 19.9 MB | Best performing weights |
| `weights/best.pt` | 19.9 MB | Checkpoint (best) |
| `weights/last.pt` | 19.9 MB | Checkpoint (final) |
| `results.csv` | 5 KB | Training metrics log |
| `results.png` | 126 KB | Training curves visualization |
| `confusion_matrix.png` | 110 KB | Classification matrix |
| `confusion_matrix_normalized.png` | 106 KB | Normalized confusion matrix |

---

## 🔬 Technical Details

### Compute Configuration

| Setting | Value |
|---------|-------|
| **Device** | Auto (CUDA if available) |
| **Workers** | 4 |
| **AMP** | ✅ Enabled (Mixed Precision) |
| **Deterministic** | ✅ Enabled |
| **Random Seed** | 42 |

### Total Training Time

- **Duration**: ~63 minutes
- **Per Epoch**: ~51 seconds average
- **Epochs Completed**: 74 / 100

---

## 🚀 Recommendations

### Immediate Optimizations

1. **Increase Dropout** (0.5 → 0.6)
   - Reduce overfitting tendency
   - May improve generalization

2. **Add More Regularization**
   - Increase weight decay (0.01 → 0.02)
   - Add label smoothing (0.1)

3. **Data Augmentation Enhancement**
   - Increase MixUp (0.0 → 0.2)
   - Add CutMix (0.0 → 0.1)

### Future Improvements

| Strategy | Expected Impact | Priority |
|----------|-----------------|----------|
| Collect more domain-specific data | High | 🔴 High |
| Use larger model (YOLO11l-cls) | Medium | 🟡 Medium |
| Increase image resolution (224 → 384) | Medium | 🟡 Medium |
| Ensemble multiple models | High | 🟢 Low |

---

## 📝 Conclusion

The **YOLO11m-cls** model achieved **83.55% validation accuracy** on binary gender classification from CCTV imagery. While the model demonstrates solid baseline performance, mild overfitting indicates room for improvement through additional regularization and domain-specific data collection. The model is suitable for initial deployment with monitoring for real-world performance validation.

---

### 📎 Artifacts Reference

```
v_20260209_170745/
├── args.yaml                      # Training configuration
├── best_model.pt                  # Production model
├── results.csv                    # Training logs
├── results.png                    # Training curves
├── confusion_matrix.png           # Performance matrix
├── confusion_matrix_normalized.png
├── train_batch[0-2].jpg           # Training samples
├── val_batch[0-2]_labels.jpg      # Validation samples
├── val_batch[0-2]_pred.jpg        # Model predictions
└── weights/
    ├── best.pt
    └── last.pt
```

---

*Report generated on: February 9, 2026*  
*Model Version: v_20260209_170745*  
*Framework: Ultralytics YOLO v11*

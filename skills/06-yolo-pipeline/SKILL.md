---
name: yolo-pipeline
description: YOLO detection, segmentation, classification, and pose estimation setup, training workflow, evaluation metrics, and XAI verification
version: 1.0.0
tags: [yolo, ultralytics, detection, segmentation, classification, deep-learning, mAP, IoU]
---

# YOLO Pipeline

## When to Use This Skill

- Detecting objects with bounding boxes in images or video
- Segmenting objects at pixel level
- Classifying entire images into categories
- Estimating human body pose (keypoints)
- Training custom YOLO models on domain-specific data
- Evaluating model performance with IoU/mAP metrics

## Decision Framework

### Which YOLO Task?

```
What output do you need?
├── "What objects are here and WHERE?" (bounding boxes)
│   └── ✅ Detection — model('img.jpg') with yolov8n.pt
│
├── "Exact pixel-level shape of each object"
│   └── ✅ Segmentation — model('img.jpg') with yolov8n-seg.pt
│
├── "What IS this image overall?" (single label)
│   └── ✅ Classification — model('img.jpg') with yolov8n-cls.pt
│
├── "What pose is this person in?" (17 keypoints)
│   └── ✅ Pose Estimation — model('img.jpg') with yolov8n-pose.pt
│
└── "Rotated/angled objects" (ships, aircraft)
    └── ✅ Oriented Bounding Boxes (OBB) — yolov8n-obb.pt
```

### Semantic vs Instance Segmentation

| Type | Question | Output | Can Count Individuals? |
|------|----------|--------|------------------------|
| **Semantic** | "What’s in the scene?" | One color per class | ❌ No (5 sheep = one green blob) |
| **Instance** (YOLO) | "Which objects where?" | Unique ID per object | ✅ Yes (5 sheep = 5 different colors) |

> **YOLO does Instance Segmentation** — each object gets its own mask and identity.

### Pre-trained Dataset Reference

| Dataset | Source | Classes | Typical Use |
|---------|--------|---------|-------------|
| COCO | Microsoft | 80 | General object detection |
| ImageNet | Stanford | 1000 | Image classification |
| DOTAv1 | Wuhan Univ. | 15 | Aerial/satellite OBB |

### When Classic CV Fails → Use Deep Learning

| Condition | Classic CV | YOLO/DL |
|-----------|-----------|----------|
| High contrast, uniform light | ✅ Works | Overkill |
| Shadows, uneven lighting | ❌ Fails | ✅ Robust |
| Touching/overlapping objects | ❌ Fails | ✅ Handles |
| Diverse viewpoints | ❌ Unreliable | ✅ Generalizes |
| Need to detect 80+ classes | ❌ Impractical | ✅ Built-in |

## YOLO Detection — The 3-Step Pipeline

### Step 1: Grid Division (Localization)
- Image is divided into grid cells (e.g., 7×7)
- Each cell is responsible for detecting objects whose **center falls within it**
- "You Only Look Once" — each cell handles its own region

### Step 2: Anchor Box Prediction
- Each cell generates **Anchor Boxes** of different aspect ratios
- For each anchor: confidence score + class probabilities + bbox coordinates
- YOLO predicts: `[center_x, center_y, width, height]` (normalized 0-1)

### Step 3: Non-Maximum Suppression (NMS)
- Multiple overlapping boxes around the same object → keep only the best one
- Low-confidence boxes are filtered first
- Among remaining overlapping boxes, NMS keeps the highest-confidence box

## Training Workflow

### Data Preparation Checklist

1. **Collect images** — diverse angles, lighting, backgrounds
2. **Label data** — Tools: makesense.ai (free, web-based), CVAT, Roboflow
3. **YOLO label format** (one `.txt` per image):
   ```
   [class_index] [center_x] [center_y] [width] [height]
   Example: 1 0.45 0.60 0.30 0.40
   ```
   All coordinates normalized to 0-1 range.
4. **Classification: no labels needed** — folder name IS the label
   ```
   dataset/
   ├── train/
   │   ├── COVID/       ← folder name = class label
   │   └── Normal/
   └── val/
       ├── COVID/
       └── Normal/
   ```
5. **Data Augmentation** — use Albumentations (auto-adjusts bbox/polygon coords)

### Key Training Parameters

| Parameter | What It Controls | Guidance |
|-----------|-----------------|----------|
| **Learning Rate** | Step size for weight updates | Too high = overshoots, too low = slow convergence |
| **Epochs** | Full passes through dataset | More is not always better (overfitting risk) |
| **Batch Size** | Images per training step | Limited by GPU memory. 16-32 typical |
| **imgsz** | Input image resolution | 640 default. Higher = better accuracy, slower |

### Transfer Learning

Don't train from scratch. Use pre-trained weights:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Pre-trained on COCO
model.train(data='my_data.yaml', epochs=100, imgsz=640)
```

> **Transfer Learning = standing on giants' shoulders.** The model already knows edges, textures, shapes from millions of images. You only retrain the final layers for your specific task.

### Saved Weights

| File | Contains | Use For |
|------|----------|--------|
| `best.pt` | Weights at **best validation accuracy** | ✅ Production / inference |
| `last.pt` | Weights at final epoch | Resume interrupted training only |

> **Always use `best.pt` for inference.** `last.pt` may have started overfitting.

## Evaluation Metrics

### Confusion Matrix Components

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actually Positive** | TP (correct detection) | FN (missed detection — WORST in medical) |
| **Actually Negative** | FP (false alarm) | TN (not used in object detection) |

> **TN is not used in object detection systems** — "correctly identifying nothing" is not meaningful.

### IoU (Intersection over Union)

```
IoU = Area of Overlap / Area of Union
```

- Range: 0 (no overlap) to 1 (perfect match)
- **IoU > 0.50** = detection counts as True Positive (standard threshold)
- Higher IoU thresholds = stricter evaluation

### mAP (Mean Average Precision)

| Metric | Threshold | Strictness |
|--------|-----------|------------|
| mAP@0.50 | 50% overlap required | Standard |
| mAP@0.50:0.95 | Average across 50%-95% | Very strict, comprehensive |

### Precision, Recall, F1

```
Precision = TP / (TP + FP)    — "Of all detections, how many are correct?"
Recall    = TP / (TP + FN)    — "Of all real objects, how many did we find?"
F1        = 2 × (P × R) / (P + R)  — Harmonic mean
```

### Overfitting Check

If training loss keeps dropping but validation loss plateaus or rises → **overfitting**. Solutions:
- More data augmentation
- Fewer epochs
- Regularization (dropout, weight decay)

## XAI (Explainable AI) — EigenCAM

DL models are **black boxes** — they give answers but don't explain why.

| Heatmap Color | Meaning |
|---------------|--------|
| 🔴 Hot (Red/Yellow) | Model's decision focus — high attention |
| 🔵 Cold (Blue/Purple) | Irrelevant to model's decision |

**Why it matters in medicine:**
- Verify the model looks at the **lesion**, not the **hospital label** in the corner
- Prove to clinicians that the AI decision is based on actual pathology
- Required for regulatory approval in many jurisdictions

## Common Parameters

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model(
    source='image.jpg',    # or '0' for webcam, or video path
    show=True,             # display results
    save=True,             # save to disk
    conf=0.25,             # minimum confidence threshold
)
```

## Industrial Lighting Systems

| Light Type | Setup | Best For |
|-----------|-------|----------|
| Array/Screen | Front-facing flat | Surface defects |
| Ring Light | Around camera lens | Small parts, uniform light |
| Back Light | Behind object | Silhouette, hole detection |
| Bar Light | Angled beam | Directional surface inspection |
| Dome Light | Surrounds object | Eliminating reflections on metal/shiny surfaces |

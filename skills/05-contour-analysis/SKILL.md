---
name: contour-analysis
description: Shape classification through mathematical ratios, bounding box extraction, centroid computation, and medical morphometric analysis
version: 1.0.0
tags: [opencv, contours, shape-analysis, features, medical-imaging, morphometry]
---

# Contour Analysis

## When to Use This Skill

- Counting objects in an image
- Measuring object area, perimeter, or physical dimensions
- Classifying shapes (round vs elongated, regular vs irregular)
- Medical image analysis (tumor shape assessment)
- Drawing bounding boxes or centroids on detected objects

## Decision Framework

### Shape Metric Selection

| Metric | Formula | Range | Tells You |
|--------|---------|-------|----------|
| **Aspect Ratio** | width / height | 0→∞ | Shape elongation (1.0 = square/circle) |
| **Extent** | Object Area / Bounding Box Area | 0→1.0 | How much the box is filled |
| **Solidity** | Object Area / Convex Hull Area | 0→1.0 | Surface regularity (1.0 = smooth, <1.0 = irregular) |
| **Eccentricity** | Minor Axis / Major Axis | 0→1.0 | Circularity (0 = circle, 1 = line) |

### Medical Application — Tumor Shape Classification

```
Solidity value?
├── ≈ 1.0 (smooth, convex surface)
│   └── Likely BENIGN — regular, well-defined boundary
│
└── << 1.0 (irregular, spiculated surface)
    └── Likely MALIGNANT — irregular projections, infiltrative margin
```

> **Solidity is a gold-standard feature in medical image analysis** for distinguishing benign vs malignant masses. Convex Hull wraps the object tightly — if the actual area is much smaller than the hull, the surface has indentations/spikes (suspicious morphology).

### Contour Finding Parameters

| Parameter | Recommended | Alternative | When |
|-----------|------------|-------------|------|
| **Mode** | `RETR_EXTERNAL` | `RETR_TREE` | External = outer boundaries only. Tree = nested hierarchy |
| **Method** | `CHAIN_APPROX_SIMPLE` | `CHAIN_APPROX_NONE` | Simple = corner points only (saves memory). None = all boundary pixels |

## Critical Gotchas

### 1. Image MUST Be Binary Before findContours

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
```

### 2. findContours Expects White Objects on Black Background

If your objects are dark on a light background, use `THRESH_BINARY_INV` to invert.

### 3. drawContours Index: -1 = All

```python
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)  # All contours
cv2.drawContours(img, contours, 0, (0, 255, 0), 2)   # Only first contour
```

### 4. Minimum Points for fitEllipse

`cv2.fitEllipse()` requires at least **5 points** in the contour. Filter small contours first.

### 5. Physical Measurement Requires Calibration

Pixel area alone is meaningless in real units. You need:
```
Real Area (mm²) = Pixel Area × (Physical Size of 1 Pixel in mm)²
```

## Quick Reference

### Complete Contour Analysis Pipeline

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Aspect Ratio
    aspect_ratio = w / h
    
    # Extent
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0
    
    # Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Centroid
    M = cv2.moments(cnt)
    if M['m00'] > 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
```

### Eccentricity via Ellipse Fitting

```python
if len(cnt) >= 5:
    ellipse = cv2.fitEllipse(cnt)
    (center, (minor_axis, major_axis), angle) = ellipse
    eccentricity = minor_axis / major_axis  # 0=circle, 1=line
```

### Scikit-Image Alternative (37 Features in One Call)

```python
from skimage.measure import label, regionprops

labeled = label(binary_image)
for region in regionprops(labeled):
    print(region.area, region.perimeter, region.solidity,
          region.eccentricity, region.centroid, region.bbox)
```

> **skimage.regionprops** extracts 37 morphometric features per object in a single call, compared to manual per-feature computation in OpenCV.

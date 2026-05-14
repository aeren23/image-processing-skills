---
name: thresholding-strategy
description: Decision matrix for Global vs Otsu vs Adaptive thresholding, CLAHE parameter tuning, and histogram-based preprocessing
version: 1.0.0
tags: [opencv, thresholding, histogram, CLAHE, segmentation, binary]
---

# Thresholding Strategy

## When to Use This Skill

- Converting an image to binary (black/white) for segmentation
- Separating foreground objects from background
- Preprocessing before contour detection or morphological operations
- Improving contrast on low-contrast images
- Working with images that have uneven lighting or shadows

## Decision Framework

### Which Thresholding Method?

```
Image lighting conditions?
├── Uniform lighting, good contrast
│   ├── You know the ideal threshold value
│   │   └── ✅ Global Threshold (cv2.threshold)
│   └── You don't know the ideal value
│       └── ✅ Otsu's Method (automatic optimal threshold)
│
├── Uneven lighting / shadows present
│   └── ✅ Adaptive Threshold (cv2.adaptiveThreshold)
│       ├── General use → ADAPTIVE_THRESH_MEAN_C
│       └── Text/document → ADAPTIVE_THRESH_GAUSSIAN_C (better)
│
└── Very low contrast (details invisible)
    └── First enhance contrast, THEN threshold:
        ├── Moderate enhancement → Histogram Stretching
        ├── Strong enhancement → Histogram Equalization
        └── Local enhancement → CLAHE (best for most cases)
```

### Contrast Enhancement Comparison

| Method | Function | Effect | Best For |
|--------|----------|--------|----------|
| Histogram Stretching | `cv2.normalize(NORM_MINMAX)` | Linear rescale min→0, max→255 | Natural-looking enhancement |
| Histogram Equalization | `cv2.equalizeHist()` | Aggressive CDF-based redistribution | Maximum contrast, looks artificial |
| CLAHE | `cv2.createCLAHE()` | Local adaptive equalization | Medical images, preserves local detail |

> **Default choice: CLAHE.** It provides strong enhancement without the over-amplification artifacts of global equalization.

### Global Threshold Types

| Type | Above Threshold | Below Threshold | Use Case |
|------|----------------|-----------------|----------|
| `THRESH_BINARY` | White (255) | Black (0) | Standard foreground extraction |
| `THRESH_BINARY_INV` | Black (0) | White (255) | Dark objects on light background |
| `THRESH_TRUNC` | Clamped to threshold | Unchanged | Brightness capping |
| `THRESH_TOZERO` | Unchanged | Set to 0 | Keep only bright regions |
| `THRESH_TOZERO_INV` | Set to 0 | Unchanged | Keep only dark regions |

## Critical Gotchas

### 1. Always Convert to Grayscale First

Thresholding only works on single-channel (grayscale) images.

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
```

### 2. Otsu Automatically Finds the Best Threshold

Otsu minimizes intra-class variance to find the optimal split point. Pass `0` as threshold and add the flag:

```python
# Otsu finds optimal threshold automatically
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### 3. Adaptive blockSize MUST Be Odd

```python
# CRASHES
cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                      cv2.THRESH_BINARY, 10, 3)  # blockSize=10 → Error!

# CORRECT
cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                      cv2.THRESH_BINARY, 11, 3)  # blockSize=11 ✓
```

### 4. CLAHE Parameters — Tuning Guide

```python
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
result = clahe.apply(gray)
```

| Parameter | Range | Effect |
|-----------|-------|--------|
| `clipLimit` | 2.0 - 3.0 (typical) | Higher = more contrast, but more noise amplification |
| `tileGridSize` | (4,4) to (16,16) | Smaller tiles = more local adaptation, risk of artifacts |

> **Safe defaults:** `clipLimit=2.5`, `tileGridSize=(8,8)`

### 5. Histogram Equalization Only Works on Grayscale

`cv2.equalizeHist()` accepts only single-channel images. For color images, convert to LAB or YCrCb, equalize the L/Y channel only, then convert back.

### 6. Reading a Histogram for Quick Diagnostics

| Histogram Shape | Diagnosis | Action |
|-----------------|-----------|--------|
| Clustered on left (dark) | Underexposed image | Apply CLAHE or equalization |
| Clustered on right (bright) | Overexposed image | Apply CLAHE or normalize |
| Narrow peak in center | Low contrast | Histogram stretching or CLAHE |
| Wide, uniform spread | Good contrast | Ready for thresholding |
| Two distinct peaks (bimodal) | Clear foreground/background | Otsu will work perfectly |

## Quick Reference

### Histogram Computation

```python
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
# [gray]    — input image (list)
# [0]       — channel index
# None      — no mask
# [256]     — number of bins
# [0, 256]  — pixel value range
```

### Complete Contrast Enhancement Pipeline

```python
# Method 1: Histogram Stretching (natural)
stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

# Method 2: Global Equalization (aggressive)
equalized = cv2.equalizeHist(gray)

# Method 3: CLAHE (recommended)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
```

### Adaptive Thresholding

```python
# MEAN_C: simple average of neighborhood
binary = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)

# GAUSSIAN_C: weighted average (better for text)
binary = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
# blockSize=11: neighborhood size (must be odd)
# C=3: constant subtracted from computed mean
```

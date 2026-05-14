---
name: morphology-toolkit
description: Opening vs Closing decision guide, structuring element selection, Top-Hat/Bottom-Hat contrast filters, and grayscale morphology
version: 1.0.0
tags: [opencv, morphology, erosion, dilation, opening, closing, binary]
---

# Morphology Toolkit

## When to Use This Skill

- Cleaning binary images after thresholding (removing noise, filling holes)
- Separating touching objects in segmentation results
- Extracting object boundaries or skeletons
- Enhancing contrast for specific features (Top-Hat/Bottom-Hat)
- Processing fingerprints, cell images, document scans

## Decision Framework

### The Core Four Operations

```
What do you need to clean up?
├── Small white noise specks on black background (external noise)
│   └── ✅ Opening (Erosion → Dilation)
│
├── Small black holes inside white objects (internal gaps)
│   └── ✅ Closing (Dilation → Erosion)
│
├── Objects touching each other (need separation)
│   └── ✅ Erosion (shrinks objects apart)
│
└── Broken lines or gaps in object boundaries
    └── ✅ Dilation (expands and connects)
```

### Opening vs Closing — The Memory Trick

| Operation | Formula | Removes | Keeps | Memory Aid |
|-----------|---------|---------|-------|------------|
| **Opening** | Erode → Dilate | External noise | Object integrity | "Opens" gaps between noise and object |
| **Closing** | Dilate → Erode | Internal holes | Object shape | "Closes" holes inside object |

> **Golden rule:** Opening cleans the **outside**, Closing fills the **inside**.

**Real-world example — Fingerprint processing:**
1. **Opening** removes background dirt/smudges
2. **Closing** reconnects broken ridge lines

### Structuring Element Selection

| Shape | OpenCV Constant | Best For |
|-------|----------------|----------|
| Rectangle/Square | `cv2.MORPH_RECT` | Angular objects, text characters |
| Ellipse/Disk | `cv2.MORPH_ELLIPSE` | Round objects, cells, coins |
| Cross (+) | `cv2.MORPH_CROSS` | Thin lines, intersections |

**Size selection rule:** The structuring element must be:
- **Larger** than the noise you want to remove
- **Smaller** than the objects you want to keep

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
```

### Contrast Enhancement Morphology

| Filter | Formula | Reveals | Use Case |
|--------|---------|---------|----------|
| **Top-Hat** | Original - Opening | Bright details on dark background | Bright spots, text on dark surface |
| **Bottom-Hat** | Closing - Original | Dark details on bright background | Dark spots, stains on light surface |

```python
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
```

## Critical Gotchas

### 1. Order Matters (Opening ≠ Closing)

Swapping the order of erosion and dilation produces opposite results:
- Erode first, then Dilate = Opening (removes small white blobs)
- Dilate first, then Erode = Closing (fills small black holes)

### 2. Grayscale Morphology Works Differently

In binary images, erosion/dilation work with set theory. In grayscale:

| Operation | Grayscale Behavior | Visual Effect |
|-----------|-------------------|---------------|
| **Dilation** | Maximum filter (picks brightest neighbor) | Image brightens, bright regions expand |
| **Erosion** | Minimum filter (picks darkest neighbor) | Image darkens, dark regions expand |

### 3. Iteration Count Amplifies Effect

```python
# Single pass
eroded = cv2.erode(binary, kernel, iterations=1)

# Multiple passes = stronger effect (equivalent to larger kernel)
eroded = cv2.erode(binary, kernel, iterations=3)
```

### 4. Morphological Gradient = Edge Extraction

```python
# Dilation - Erosion = object boundary
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
```

## Quick Reference

### Basic Operations

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

eroded = cv2.erode(binary, kernel, iterations=1)
dilated = cv2.dilate(binary, kernel, iterations=1)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
```

### Advanced Operations

```python
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)  # Boundary
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)      # Bright details
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)   # Dark details
```

### Common Processing Chain

```python
# Typical binary cleanup pipeline
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # Remove noise
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel) # Fill holes
```

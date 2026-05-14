---
name: preprocessing-decisions
description: Filter selection decision tree, noise identification, edge detection priority, and kernel parameter rules
version: 1.0.0
tags: [opencv, filtering, blur, edge-detection, noise, convolution]
---

# Preprocessing Decisions

## When to Use This Skill

- Choosing a blur/smoothing filter for noise reduction
- Selecting an edge detection algorithm
- Identifying noise type in an image
- Setting kernel sizes and filter parameters
- Building a preprocessing pipeline before segmentation or detection

## Decision Framework

### Filter Selection Decision Tree

```
What type of noise?
├── Salt & Pepper (random black/white dots)
│   └── ✅ Median Filter (cv2.medianBlur) — BEST IN THE WORLD for this
│
├── Gaussian noise (camera sensor heat, general grain)
│   └── ✅ Gaussian Blur (cv2.GaussianBlur)
│
├── Unknown noise + must preserve edges
│   └── ✅ Bilateral Filter (cv2.bilateralFilter) — kills noise, keeps edges
│
├── General smoothing (no specific noise type)
│   └── ✅ Mean Filter (cv2.blur) — simplest, fastest
│
└── Medical image with bias field / Rician noise
    └── ✅ Non-Local Means (cv2.fastNlMeansDenoising)
```

### Filter Comparison Matrix

| Filter | Speed | Edge Preservation | Noise Removal | Best For |
|--------|-------|-------------------|---------------|----------|
| Mean (`cv2.blur`) | ⚡⚡⚡ | ❌ Poor | ⭐⭐ | General smoothing |
| Gaussian (`cv2.GaussianBlur`) | ⚡⚡⚡ | ⭐ Fair | ⭐⭐⭐ | Gaussian noise, pre-Canny |
| Median (`cv2.medianBlur`) | ⚡⚡ | ⭐⭐ Good | ⭐⭐⭐⭐⭐ (S&P) | Salt & Pepper noise |
| Bilateral (`cv2.bilateralFilter`) | ⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Edge-aware denoising |

> **Rule of thumb:** If you don't know the noise type, start with Gaussian. If edges matter, use Bilateral. If you see random black/white dots, use Median — nothing else comes close.

### Edge Detection Priority

```
What do you need to detect?
├── General edges (most use cases)
│   └── ✅ Canny (cv2.Canny) — gold standard, 5-step pipeline
│
├── Directional edges (horizontal OR vertical)
│   └── ✅ Sobel (cv2.Sobel) — first derivative, specify dx/dy
│
├── Fine detail + corners + all boundaries
│   └── ✅ Laplacian (cv2.Laplacian) — second derivative, zero-crossing
│
└── Text/document character edges (OCR preprocessing)
    └── ✅ Prewitt — better than Sobel for text sharpness
```

### Edge Detection Comparison

| Detector | Derivative | Output | Strengths | Weaknesses |
|----------|-----------|--------|-----------|------------|
| Sobel | 1st | Directional gradient map | Clean directional edges | Misses some corners |
| Prewitt | 1st | Similar to Sobel | Better for text/documents | Noisier than Sobel |
| Laplacian | 2nd | All edges via zero-crossing | Catches finest details | Very noise-sensitive |
| Canny | Multi-step | Thin binary edges | Best general-purpose | Sensitive to parameters |

## Critical Gotchas

### 1. Gaussian Blur Before Canny — MANDATORY

Canny is extremely noise-sensitive. Without pre-blurring, it will detect noise as edges.

```python
# WRONG — will produce noisy edges
edges = cv2.Canny(img, 100, 200)

# CORRECT — always blur first
blurred = cv2.GaussianBlur(img, (5, 5), 0)
edges = cv2.Canny(blurred, 100, 200)
```

### 2. Kernel Size Must ALWAYS Be Odd

Every kernel/filter size in OpenCV must be an odd number (3, 5, 7, 9...). Even numbers will crash.

```python
# CRASHES
cv2.GaussianBlur(img, (4, 4), 0)  # Error!

# CORRECT
cv2.GaussianBlur(img, (5, 5), 0)
```

### 3. Bilateral Filter Is Slow

Bilateral preserves edges beautifully but is **significantly slower** than other filters. For real-time video, prefer Gaussian or Median.

```python
# Bilateral parameters: (src, diameter, sigmaColor, sigmaSpace)
cv2.bilateralFilter(img, 9, 75, 75)
# diameter=9: neighborhood size
# sigmaColor=75: color range for blending
# sigmaSpace=75: spatial distance for blending
```

### 4. Median Filter Kernel Must Be a Single Odd Integer

Unlike other filters that take a tuple `(5, 5)`, median takes just one integer:

```python
# WRONG
cv2.medianBlur(img, (5, 5))  # Error!

# CORRECT
cv2.medianBlur(img, 5)
```

### 5. The ddepth=-1 Convention

In filter functions, `ddepth=-1` means "output same depth as input." For float precision:

```python
cv2.filter2D(img, -1, kernel)        # Output = same type as input
cv2.filter2D(img, cv2.CV_64F, kernel) # Output = 64-bit float
```

## Quick Reference

### Convolution Basics

- **Kernel (Mask):** Small matrix (3×3, 5×5) slid over the image
- **Convolution:** Multiply kernel × image patch, sum results, write to center pixel
- **Padding:** Add zero-pixels around borders so kernel can process edge pixels
- Larger kernel = stronger effect but slower and may lose detail

### Frequency Domain Concepts

| Frequency | Visual Appearance | Examples |
|-----------|-------------------|----------|
| Low frequency | Smooth, gradual changes | Background, skin, sky |
| High frequency | Sharp, sudden changes | Edges, textures, noise |

- **Low-pass filters** (blur) remove high frequency → smooth image
- **High-pass filters** (sharpen) emphasize high frequency → enhance edges
- **Sharpening kernel example:** Center = high positive (e.g., 9), neighbors = negative (e.g., -1)

### Noise Type Identification

| Noise Type | Visual Pattern | Cause | Best Filter |
|------------|---------------|-------|-------------|
| Salt & Pepper | Random pure white and pure black pixels | Sensor errors, transmission | Median |
| Gaussian | Uniform grain/static across image | Sensor heat, low light | Gaussian Blur |
| Speckle | Multiplicative granular noise | Ultrasound, SAR radar | Bilateral |
| MRI Bias Field | Smooth intensity variation across image | B0 field inhomogeneity | Non-Local Means |

### Canny Edge Detection — The 5 Steps

1. **Gaussian Blur** — Remove noise (you should also do this before calling Canny)
2. **Sobel Gradients** — Compute intensity gradients in X and Y
3. **Gradient Magnitude & Direction** — Find edge strength and angle
4. **Non-Maximum Suppression** — Thin edges to 1-pixel width
5. **Hysteresis Thresholding** — Connect edges using upper/lower thresholds

```python
# threshold1 = lower bound, threshold2 = upper bound
# Ratio recommendation: 1:2 or 1:3
edges = cv2.Canny(blurred, 50, 150)  # 1:3 ratio
```

---
name: image-fundamentals
description: Color space decisions, format selection, coordinate conventions, and bit depth trade-offs for image processing
version: 1.0.0
tags: [opencv, color-space, image-format, coordinates, fundamentals]
---

# Image Fundamentals

## When to Use This Skill

- Starting any new image processing pipeline
- Choosing between color spaces (BGR, RGB, HSV, Gray, LAB)
- Deciding image format for saving/loading
- Debugging coordinate-related bugs in OpenCV
- Calculating memory requirements for image data

## Decision Framework

### Color Space Selection

| Goal | Convert To | OpenCV Code | Why |
|------|-----------|-------------|-----|
| Display with matplotlib | RGB | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` | Matplotlib expects RGB, OpenCV loads BGR |
| Grayscale processing | Gray | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` | Single channel, faster computation |
| Color-based object filtering | HSV | `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)` | Isolate hue independently from brightness |
| Perceptual color difference | LAB | `cv2.cvtColor(img, cv2.COLOR_BGR2LAB)` | L=lightness, A/B=color, perceptually uniform |
| Print/publishing output | CMYK | External library | Subtractive color model for ink |

### Format Selection

| Format | Use When | Compression | Quality Loss |
|--------|----------|-------------|-------------|
| `.jpg/.jpeg` | Web, general photos | Lossy | Yes — artifacts at low quality |
| `.png` | Transparency needed, lossless required | Lossless | No |
| `.tiff` | Medical/scientific, archival | Both | Depends on setting |
| `.dcm` (DICOM) | Clinical X-ray, MR, CT | Lossless | No |
| `.nii` (NIfTI) | Neuroimaging (brain MRI) | Lossless | No |
| `.gif` | Animations, indexed color | Lossless (256 colors) | Color palette limited |

### Bit Depth Decision

| Bit Depth | Colors | Use Case | Memory per Pixel |
|-----------|--------|----------|------------------|
| 1-bit | 2 (B/W) | Binary masks, documents | 0.125 bytes |
| 8-bit gray | 256 shades | Standard grayscale processing | 1 byte |
| 24-bit (3×8) | 16.7M | Standard color (BGR/RGB) | 3 bytes |
| 32-bit float | Continuous | HDR, scientific computation | 4 bytes |
| 48-bit (3×16) | 281T | Medical, raw camera | 6 bytes |

**Memory formula:** `width × height × channels × bytes_per_channel`

Example: 1920×1080 RGB (8-bit) = 1920 × 1080 × 3 × 1 = **6,220,800 bytes ≈ 5.93 MB** in RAM

> **Note:** Compressed file size (JPG/PNG on disk) is much smaller, but OpenCV always decompresses to full size in memory.

## Critical Gotchas

### 1. The BGR Trap (Most Common OpenCV Bug)

OpenCV loads images in **BGR** order, not RGB. If you display with matplotlib or send to a model expecting RGB, colors will be swapped (blue shirt appears red).

```python
# WRONG — colors will be inverted
plt.imshow(cv2.imread('photo.jpg'))

# CORRECT
img = cv2.imread('photo.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
```

### 2. The Triple Coordinate Convention

OpenCV uses THREE different coordinate orders depending on context:

| Operation | Convention | Example |
|-----------|------------|----------|
| Array indexing | `[y, x]` (row, col) | `pixel = img[200, 150]` |
| Drawing functions | `(x, y)` (col, row) | `cv2.circle(img, (150, 200), 5, ...)` |
| Resize function | `(width, height)` = `(x, y)` | `cv2.resize(img, (640, 480))` |
| Shape property | `(height, width, channels)` | `h, w, c = img.shape` |

> **This is the #1 source of coordinate bugs.** When debugging spatial issues, always check which convention the function expects.

### 3. Grayscale Reading Shortcut

```python
# These are equivalent:
gray = cv2.cvtColor(cv2.imread('img.jpg'), cv2.COLOR_BGR2GRAY)
gray = cv2.imread('img.jpg', 0)  # 0 flag = grayscale directly
```

### 4. Data Type Is Always uint8

OpenCV images default to `uint8` (0-255). If you do math that overflows:
- OpenCV functions (`cv2.add`): **clips** to 0-255 (safe)
- NumPy operations (`img1 + img2`): **wraps around** (255+1=0, dangerous!)

```python
# SAFE — clips at 255
result = cv2.add(img1, img2)

# DANGEROUS — wraps around
result = img1 + img2  # 200 + 100 = 44 (not 255!)
```

## Quick Reference

### Image I/O

```python
img = cv2.imread('file.jpg')          # Load BGR
img = cv2.imread('file.jpg', 0)       # Load grayscale
cv2.imwrite('out.png', img)           # Save (format from extension)
cv2.imshow('Window', img)             # Display
cv2.waitKey(0)                        # Wait for keypress
cv2.destroyAllWindows()               # Clean up windows
```

### ROI (Region of Interest)

```python
roi = img[y1:y2, x1:x2]              # Crop — note: [rows, cols]
img[y1:y2, x1:x2] = 0                # Set region to black
```

### Shape and Size

```python
h, w, c = img.shape                   # Height, Width, Channels
total_pixels = img.size                # Total elements
dtype = img.dtype                      # Usually uint8
```

### HSV Value Ranges in OpenCV

| Channel | Range | Notes |
|---------|-------|-------|
| H (Hue) | 0-179 | NOT 0-360! Divided by 2 to fit uint8 |
| S (Saturation) | 0-255 | 0=gray/pastel, 255=vivid |
| V (Value/Brightness) | 0-255 | 0=black, 255=brightest |

**Common HSV hue ranges:**
- Red: 0-15 AND 160-179 (wraps around!)
- Green: 45-90
- Blue: 105-135
- Yellow: 25-45

> **Red wraps around the hue circle.** You need TWO masks combined with OR to detect red objects in HSV.

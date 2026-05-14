# Pipeline: Medical X-Ray Preprocessing

A step-by-step preprocessing chain for medical radiographs (X-ray, CT, MRI) before analysis or model inference.

## Pipeline Flow

```
Input Image
    │
    ▼
[1] Convert to Grayscale
    │
    ▼
[2] CLAHE Enhancement (clipLimit=2.5, tileGrid=8×8)
    │
    ▼
[3] Gaussian Blur (5×5) — remove sensor noise
    │
    ▼
[4] Adaptive Threshold (GAUSSIAN_C, blockSize=11, C=3)
    │
    ▼
[5] Morphology: Opening (ELLIPSE 3×3) — remove external noise
    │
    ▼
[6] Morphology: Closing (ELLIPSE 5×5) — fill internal gaps
    │
    ▼
Clean Binary Output
```

## Code

```python
import cv2
import numpy as np

def preprocess_medical_xray(image_path):
    # 1. Load as grayscale
    gray = cv2.imread(image_path, 0)
    
    # 2. CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 3. Gaussian blur to suppress sensor noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # 4. Adaptive threshold for uneven illumination
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 3
    )
    
    # 5. Opening: remove small external noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    
    # 6. Closing: fill small internal holes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    result = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
    
    return result
```

## Parameter Tuning Guide

| Parameter | Default | Adjust When |
|-----------|---------|-------------|
| CLAHE clipLimit | 2.5 | Increase for very dark images, decrease if noise amplified |
| CLAHE tileGridSize | (8,8) | Smaller tiles = more local detail, larger = smoother |
| GaussianBlur kernel | (5,5) | Larger for noisier images, smaller to preserve fine detail |
| Adaptive blockSize | 11 | Increase for larger structures, decrease for fine detail |
| Adaptive C | 3 | Increase to suppress more background, decrease to keep faint features |
| Opening kernel | 3×3 | Match to noise speck size |
| Closing kernel | 5×5 | Match to hole/gap size |

## Why This Order?

1. **Grayscale first** — thresholding requires single channel
2. **CLAHE before blur** — enhance contrast while detail still exists
3. **Blur before threshold** — prevents noise from creating false binary regions
4. **Adaptive over Otsu** — medical images often have uneven exposure
5. **Opening before Closing** — clean noise first, then repair structures

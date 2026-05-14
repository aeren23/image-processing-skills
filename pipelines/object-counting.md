# Pipeline: Object Counting

Count distinct objects in an image using classical computer vision (no deep learning required).

## Pipeline Flow

```
Input Image
    │
    ▼
[1] Convert to Grayscale
    │
    ▼
[2] Gaussian Blur (5×5)
    │
    ▼
[3] Otsu Threshold (automatic)
    │
    ▼
[4] Morphology: Opening (3×3) — separate touching objects
    │
    ▼
[5] Find Contours (RETR_EXTERNAL)
    │
    ▼
[6] Filter by Area — remove tiny noise contours
    │
    ▼
[7] Count + Draw Bounding Boxes
    │
    ▼
Annotated Output + Count
```

## Code

```python
import cv2
import numpy as np

def count_objects(image_path, min_area=500):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, binary = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, str(count), (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return img, count
```

## Key Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Otsu over manual threshold | Automatic optimal split | No guesswork needed |
| BINARY_INV | Objects become white | findContours expects white-on-black |
| RETR_EXTERNAL | Outer contours only | Inner holes don't count as objects |
| Area filter | Skip tiny contours | Noise contours produce false counts |

## When This Pipeline Fails

- **Touching objects** → Add watershed segmentation or increase opening
- **Uneven lighting** → Replace Otsu with Adaptive Threshold
- **Complex scenes** → Switch to YOLO detection

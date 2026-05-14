# Pipeline: Color Object Tracking

Track a specific colored object (e.g., red ball, blue cap) in real-time video using HSV color space filtering.

## Pipeline Flow

```
Camera Frame
    │
    ▼
[1] Convert BGR → HSV
    │
    ▼
[2] Define HSV Lower/Upper Bounds
    │
    ▼
[3] cv2.inRange() → Binary Mask
    │
    ▼
[4] Median Blur (5) — clean mask noise
    │
    ▼
[5] Dilation (3×3) — fill mask gaps
    │
    ▼
[6] bitwise_and with Original → Isolated Object
    │
    ▼
Tracked Object Output
```

## Code

```python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Blue object HSV range (adjust per object)
lower = np.array([105, 50, 50])
upper = np.array([135, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    
    # Clean the mask
    mask = cv2.medianBlur(mask, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Apply mask to original
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    cv2.imshow('Tracking', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## HSV Ranges Quick Reference

| Color | H Lower | H Upper | Note |
|-------|---------|---------|------|
| Red | 0-15 | 160-179 | ⚠️ Wraps around! Need TWO masks + OR |
| Orange | 15-25 | | |
| Yellow | 25-45 | | |
| Green | 45-90 | | |
| Blue | 105-135 | | |
| Purple | 135-160 | | |

**S and V ranges:** Typically `[50, 50]` to `[255, 255]`. Increase lower S/V to exclude washed-out or dark regions.

## Detecting Red (Special Case)

Red wraps around the hue circle. Use two ranges combined:

```python
lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([15, 255, 255])
lower_red2, upper_red2 = np.array([160, 50, 50]), np.array([179, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)
```

## Finding HSV Bounds Interactively (Trackbar)

When you don't know the exact HSV range, use trackbars to find it in real-time:

```python
cv2.createTrackbar('H_low', 'Controls', 0, 179, lambda x: None)
cv2.createTrackbar('H_high', 'Controls', 179, 179, lambda x: None)
cv2.createTrackbar('S_low', 'Controls', 50, 255, lambda x: None)
cv2.createTrackbar('S_high', 'Controls', 255, 255, lambda x: None)
cv2.createTrackbar('V_low', 'Controls', 50, 255, lambda x: None)
cv2.createTrackbar('V_high', 'Controls', 255, 255, lambda x: None)
```

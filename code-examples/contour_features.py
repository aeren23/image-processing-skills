"""Extract contour features: area, perimeter, centroid, solidity."""
import cv2
import numpy as np

img = cv2.imread('input.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area < 100:  # Skip tiny noise
        continue

    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # Centroid
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00']) if M['m00'] > 0 else 0
    cy = int(M['m01'] / M['m00']) if M['m00'] > 0 else 0

    print(f'Object {i}: area={area:.0f}, perimeter={perimeter:.1f}, '
          f'solidity={solidity:.3f}, centroid=({cx},{cy})')

    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

cv2.imshow('Contour Features', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

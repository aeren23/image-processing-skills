"""Compare blur filters: Mean vs Gaussian vs Median vs Bilateral."""
import cv2
import numpy as np

img = cv2.imread('input.jpg')

mean = cv2.blur(img, (5, 5))
gaussian = cv2.GaussianBlur(img, (5, 5), 0)
median = cv2.medianBlur(img, 5)  # Note: single int, not tuple
bilateral = cv2.bilateralFilter(img, 9, 75, 75)

cv2.imshow('Original', img)
cv2.imshow('Mean', mean)
cv2.imshow('Gaussian', gaussian)
cv2.imshow('Median', median)
cv2.imshow('Bilateral', bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()

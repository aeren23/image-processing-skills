"""Demonstrate morphological operations: Erosion, Dilation, Opening, Closing."""
import cv2
import numpy as np

gray = cv2.imread('input.jpg', 0)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

eroded = cv2.erode(binary, kernel, iterations=1)
dilated = cv2.dilate(binary, kernel, iterations=1)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # Remove noise
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Fill holes

cv2.imshow('Binary', binary)
cv2.imshow('Eroded', eroded)
cv2.imshow('Dilated', dilated)
cv2.imshow('Opened', opened)
cv2.imshow('Closed', closed)
cv2.waitKey(0)
cv2.destroyAllWindows()

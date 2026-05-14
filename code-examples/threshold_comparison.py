"""Compare thresholding methods: Global vs Otsu vs Adaptive."""
import cv2

gray = cv2.imread('input.jpg', 0)

# Global threshold (manual value)
_, global_th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Otsu (automatic optimal threshold)
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive Mean
adaptive_mean = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)

# Adaptive Gaussian (better for text)
adaptive_gauss = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)

cv2.imshow('Global (127)', global_th)
cv2.imshow('Otsu', otsu)
cv2.imshow('Adaptive Mean', adaptive_mean)
cv2.imshow('Adaptive Gaussian', adaptive_gauss)
cv2.waitKey(0)
cv2.destroyAllWindows()

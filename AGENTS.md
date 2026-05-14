# Image Processing Skills — Agent Instructions

This repository contains decision-focused skills for computer vision and image processing tasks using OpenCV, YOLO, and MediaPipe.

## How to Use These Skills

1. **Skills are in `skills/*/SKILL.md`** — each is a self-contained decision guide
2. **Pipeline recipes are in `pipelines/`** — ready-to-use processing chains
3. **Code examples are in `code-examples/`** — minimal runnable scripts

## Key Conventions

- All OpenCV code assumes **BGR** color order (not RGB)
- Coordinate convention: `img[y, x]` for array indexing, `(x, y)` for function parameters
- Kernel sizes must **always be odd** numbers (3, 5, 7...)
- Always convert to grayscale before thresholding or edge detection
- Apply Gaussian blur before Canny edge detection to prevent noise artifacts

## Skill Loading Priority

When working on an image processing task, load skills in this order:
1. **image-fundamentals** — for format/color space decisions
2. **preprocessing-decisions** — for filter selection
3. **thresholding-strategy** — if segmentation is needed
4. **morphology-toolkit** — if binary cleanup is needed
5. **contour-analysis** — if shape measurement is needed
6. **yolo-pipeline** — if deep learning detection/segmentation is needed
7. **mediapipe-tracking** — if real-time body/hand/face tracking is needed
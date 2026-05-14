---
name: mediapipe-tracking
description: Real-time face, hand, and body pose tracking with Google MediaPipe, solution selection, and comparison with YOLO pose estimation
version: 1.0.0
tags: [mediapipe, pose-estimation, hand-tracking, face-mesh, real-time, landmarks]
---

# MediaPipe Tracking

## When to Use This Skill

- Tracking human body pose in real-time video
- Detecting and tracking hand gestures or finger positions
- Building face filters, expression analysis, or gaze tracking
- Needing 3D depth information (Z-axis) for body landmarks
- Creating fitness apps, sign language recognition, or virtual avatars

## Decision Framework

### Which MediaPipe Solution?

```
What body part do you need to track?
├── Full body pose (standing, sitting, exercising)
│   └── ✅ Pose Estimation — 33 landmarks, 3D (x, y, z)
│
├── Hands and fingers
│   └── ✅ Hand Tracking — 21 landmarks per hand
│
├── Face details (lips, eyes, eyebrows, jawline)
│   └── ✅ Face Mesh — 468 landmarks, 3D
│
└── Multiple tasks simultaneously
    └── ✅ Holistic — combines Pose + Hands + Face
```

### MediaPipe vs YOLO Pose Estimation

| Feature | MediaPipe Pose | YOLO Pose |
|---------|---------------|----------|
| **Landmarks** | 33 (full body + face) | 17 (body only) |
| **Depth (Z-axis)** | ✅ Yes (camera-relative) | ❌ No (2D only) |
| **Speed** | ⚡ Very fast (optimized for mobile) | ⚡ Fast |
| **Multi-person** | ❌ Single person per frame | ✅ Multiple people |
| **3D Avatar** | ✅ Possible (Z depth) | ❌ Not possible |
| **Hand details** | ✅ 21 points per hand (separate solution) | ❌ Not available |
| **Face details** | ✅ 468 points (separate solution) | ❌ Not available |
| **Best for** | Single-user apps, gesture control, fitness | Multi-person detection, surveillance |

> **Rule of thumb:** Need detailed single-person anatomy (hands, face, depth)? → **MediaPipe.** Need to detect multiple people quickly? → **YOLO Pose.**

### Landmark Counts Reference

| Solution | Points | Notable Landmarks |
|----------|--------|--------------------|
| **Pose** | 33 | Nose, shoulders, elbows, wrists, hips, knees, ankles, eyes, ears |
| **Hand** | 21 | Wrist, thumb (4 joints), index (4), middle (4), ring (4), pinky (4) |
| **Face Mesh** | 468 | Lips contour, eye contour, eyebrows, nose bridge, jawline, iris |

## Integration Guide

### Pose Estimation Setup

```python
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5,
                   min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow('Pose', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

### Hand Tracking Setup

```python
mp_hands = mp.solutions.hands

with mp_hands.Hands(max_num_hands=2,
                     min_detection_confidence=0.7) as hands:
    # Same loop pattern as pose
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks,
                mp_hands.HAND_CONNECTIONS)
```

### Face Mesh Setup

```python
mp_face = mp.solutions.face_mesh

with mp_face.FaceMesh(max_num_faces=1,
                       min_detection_confidence=0.5) as face_mesh:
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(
                frame, face_landmarks,
                mp_face.FACEMESH_TESSELATION)
```

## Critical Gotchas

### 1. MediaPipe Expects RGB, Not BGR

MediaPipe processes RGB images. OpenCV captures BGR. Always convert:
```python
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = pose.process(rgb)
```

### 2. Landmark Coordinates Are Normalized (0-1)

Landmark `.x` and `.y` are normalized to image dimensions. To get pixel coordinates:
```python
h, w, _ = frame.shape
lm = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
pixel_x, pixel_y = int(lm.x * w), int(lm.y * h)
```

### 3. Z-Depth Is Relative, Not Absolute

The `.z` coordinate represents depth relative to the hip center (Pose) or wrist (Hand). It is NOT world-scale distance in meters.

### 4. Single Person Limitation

MediaPipe Pose processes **one person per frame**. For multi-person scenarios, use YOLO Pose or run MediaPipe per detected person crop.

### 5. Use the Context Manager

Always use `with` statement for proper resource cleanup:
```python
# CORRECT
with mp_pose.Pose() as pose:
    ...

# AVOID
pose = mp_pose.Pose()
# resources may leak
```

## Application Ideas

| Application | Solution | Key Landmarks |
|-------------|----------|----------------|
| Exercise rep counter | Pose | Shoulder, elbow, wrist angles |
| Sign language recognition | Hand | All 21 finger joints |
| Drowsiness detection | Face Mesh | Eye aspect ratio (EAR) |
| Virtual try-on | Pose + Face | Body proportions + face alignment |
| Gesture-controlled UI | Hand | Fingertip positions (index, thumb) |
| Scoliosis screening | Pose | Shoulder/hip symmetry analysis |

"""YOLO inference for detection, segmentation, and classification."""
from ultralytics import YOLO

# Detection (bounding boxes)
model_det = YOLO('yolov8n.pt')
results = model_det('input.jpg', show=True, save=True, conf=0.25)

# Segmentation (pixel-level masks)
model_seg = YOLO('yolov8n-seg.pt')
results = model_seg('input.jpg', show=True, save=True)

# Classification (single label per image)
model_cls = YOLO('yolov8n-cls.pt')
results = model_cls('input.jpg', show=True)

# Pose Estimation (17 keypoints)
model_pose = YOLO('yolov8n-pose.pt')
results = model_pose('input.jpg', show=True, save=True)

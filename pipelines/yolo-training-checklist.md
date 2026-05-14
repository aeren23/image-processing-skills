# Pipeline: YOLO Training Checklist

End-to-end checklist for training a custom YOLO model, from data collection to deployment.

## Phase 1: Data Collection

- [ ] Define classes to detect (keep it focused, 1-10 classes for v1)
- [ ] Collect 100-500+ images per class (more = better)
- [ ] Ensure diversity: angles, lighting, backgrounds, distances
- [ ] Include negative examples (images without target objects)
- [ ] Resolution: match expected inference resolution (640px default)

## Phase 2: Labeling

- [ ] Choose labeling tool:
  - **Detection:** makesense.ai (free, web-based) or CVAT
  - **Segmentation:** Roboflow or CVAT (polygon mode)
  - **Classification:** No labeling needed — use folder structure
- [ ] Create `classes.txt` with class names (one per line)
- [ ] Label all images consistently
- [ ] Export in **YOLO format**: `[class_id] [cx] [cy] [w] [h]` (normalized 0-1)
- [ ] Verify labels visually (spot-check at least 10%)

## Phase 3: Data Preparation

- [ ] Split dataset: 80% train / 20% val (or 70/20/10 with test)
- [ ] Apply data augmentation (Albumentations recommended):
  - Horizontal flip
  - Rotation (±15°)
  - Brightness/contrast variation
  - Random crop and zoom
- [ ] Create `data.yaml` configuration:
  ```yaml
  path: /path/to/dataset
  train: train/images
  val: val/images
  
  nc: 2  # number of classes
  names: ['class_a', 'class_b']
  ```

## Phase 4: Training

- [ ] Select base model size:
  | Model | Speed | Accuracy | GPU Memory |
  |-------|-------|----------|------------|
  | yolov8n | Fastest | Lowest | ~2 GB |
  | yolov8s | Fast | Good | ~4 GB |
  | yolov8m | Medium | Better | ~8 GB |
  | yolov8l | Slow | High | ~12 GB |
  | yolov8x | Slowest | Highest | ~16 GB |

- [ ] Start training with transfer learning:
  ```python
  from ultralytics import YOLO
  model = YOLO('yolov8n.pt')  # pre-trained weights
  model.train(
      data='data.yaml',
      epochs=100,
      imgsz=640,
      batch=16,
      patience=20,  # early stopping
  )
  ```
- [ ] Monitor training loss curves in TensorBoard/WandB
- [ ] Check for overfitting: val_loss should decrease alongside train_loss

## Phase 5: Evaluation

- [ ] Use `best.pt` (NOT `last.pt`) for evaluation
- [ ] Check key metrics:
  | Metric | Good | Excellent |
  |--------|------|-----------|
  | mAP@0.50 | > 0.70 | > 0.85 |
  | mAP@0.50:0.95 | > 0.45 | > 0.65 |
  | Precision | > 0.80 | > 0.90 |
  | Recall | > 0.75 | > 0.85 |
- [ ] Analyze confusion matrix for class-specific weaknesses
- [ ] Run inference on unseen test images

## Phase 6: XAI Verification (Medical/Critical)

- [ ] Generate EigenCAM heatmaps on test predictions
- [ ] Verify model focuses on actual pathology/features, not artifacts
- [ ] Document XAI findings for stakeholder review

## Phase 7: Deployment

- [ ] Export model: `model.export(format='onnx')` or keep `.pt`
- [ ] Set appropriate confidence threshold for use case:
  - Medical (minimize FN): `conf=0.15` (low threshold, catch everything)
  - General (balanced): `conf=0.25` (default)
  - High precision needed: `conf=0.50+` (only confident detections)
- [ ] Test on target hardware (edge device, server, etc.)
- [ ] Monitor performance in production

## Common Pitfalls

| Problem | Symptom | Solution |
|---------|---------|----------|
| Not enough data | Low mAP, high variance | More images + augmentation |
| Class imbalance | One class dominates | Oversample minority class |
| Overfitting | Train loss ↓, val loss ↑ | More augmentation, fewer epochs |
| Wrong image size | Poor detection of small objects | Increase imgsz (1280) |
| Label errors | Inconsistent predictions | Re-audit labels |

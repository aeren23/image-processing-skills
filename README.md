# 🖼️ Image Processing Skills for AI Agents

> Decision-focused SKILL.md files for AI coding agents working on computer vision projects.

This repository provides **opinionated decision frameworks**, **common gotchas**, and **pipeline recipes** for image processing tasks. Unlike generic OpenCV tutorials, these skills focus on **when** to use what, in **what order**, and **what traps** to avoid.

## Why This Exists

AI coding agents already know OpenCV's API. What they lack is:
- **Decision guidance**: "Which filter for this noise type?"
- **Pipeline ordering**: "What MUST come before Canny edge detection?"
- **Parameter gotchas**: "`blockSize` must be odd, `resize()` takes (x,y) not (y,x)"
- **Domain-specific rules**: "Solidity ≈ 1.0 = benign, < 1.0 = malign in tumor analysis"

## 📚 Skills

| # | Skill | Focus | Level |
|---|-------|-------|-------|
| 1 | [Image Fundamentals](skills/01-image-fundamentals/SKILL.md) | Color spaces, formats, coordinate traps | 🟢 Beginner |
| 2 | [Preprocessing Decisions](skills/02-preprocessing-decisions/SKILL.md) | Filter selection tree, noise identification | 🟡 Intermediate |
| 3 | [Thresholding Strategy](skills/03-thresholding-strategy/SKILL.md) | Global vs Otsu vs Adaptive decision matrix | 🟡 Intermediate |
| 4 | [Morphology Toolkit](skills/04-morphology-toolkit/SKILL.md) | Opening vs Closing, structuring element selection | 🟡 Intermediate |
| 5 | [Contour Analysis](skills/05-contour-analysis/SKILL.md) | Shape metrics, medical classification ratios | 🔴 Advanced |
| 6 | [YOLO Pipeline](skills/06-yolo-pipeline/SKILL.md) | Detection, segmentation, classification, XAI | 🔴 Advanced |
| 7 | [MediaPipe Tracking](skills/07-mediapipe-tracking/SKILL.md) | Face, hand, pose landmark tracking | 🔴 Advanced |

## 🔧 Pipeline Recipes

Ready-to-use processing chains:

| Recipe | Use Case |
|--------|----------|
| [Medical X-Ray Preprocessing](pipelines/medical-xray-preprocessing.md) | CLAHE → Filter → Threshold → Morph |
| [Object Counting](pipelines/object-counting.md) | Threshold → Morph → Contour → Count |
| [Color Object Tracking](pipelines/color-object-tracking.md) | HSV → inRange → Mask → Bitwise |
| [YOLO Training Checklist](pipelines/yolo-training-checklist.md) | Data → Label → Augment → Train → Evaluate |

## 💻 Code Examples

Minimal, runnable Python scripts demonstrating each skill:

| Script | Demonstrates |
|--------|--------------|
| [filter_comparison.py](code-examples/filter_comparison.py) | Mean vs Gaussian vs Median vs Bilateral |
| [threshold_comparison.py](code-examples/threshold_comparison.py) | Global vs Otsu vs Adaptive |
| [morphology_demo.py](code-examples/morphology_demo.py) | Erosion, Dilation, Opening, Closing |
| [contour_features.py](code-examples/contour_features.py) | Area, perimeter, solidity, centroid |
| [hsv_color_filter.py](code-examples/hsv_color_filter.py) | HSV masking pipeline |
| [yolo_inference.py](code-examples/yolo_inference.py) | YOLO detect/segment/classify |
| [mediapipe_pose.py](code-examples/mediapipe_pose.py) | MediaPipe pose estimation |

## 🤖 For AI Agents

This repo follows the `SKILL.md` standard. To use with your AI coding agent:

1. **Clone** this repo into your project or reference it
2. **AGENTS.md** at root provides the global context
3. Each **SKILL.md** is self-contained and can be loaded independently
4. **Pipeline recipes** provide copy-paste ready processing chains

Compatible with: Cursor, Claude Code, GitHub Copilot, Gemini CLI, and any agent that reads `.md` context files.

## 📄 License

MIT — Use freely in any project.

---

*Built by [Ali Eren](https://github.com/aeren23) — Backend .NET Developer & AI Engineering enthusiast at Pamukkale University.*
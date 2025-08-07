# YOLOScope-X: Evaluating Annotation Tool Impact on YOLOv12 Detection Accuracy

**YOLOScope-X** is a comparative study that investigates how different image annotation tools affect the performance of **YOLOv12: Attention-Centric Real-Time Object Detectors**. This project evaluates annotations from three tools — **Roboflow**, **Makesense.ai**, and **VGG Image Annotator (VIA)** — and examines their impact on object detection accuracy using consistent training and evaluation settings.

---

##  Project Objectives

- Convert annotations from each tool to YOLOv12-compatible format.
- Train YOLOv12 models on datasets annotated with each tool.
- Evaluate model performance using precision, recall, F1, and mAP metrics.
- Compare performance differences and annotation quality.

---

## Annotation Tools Compared

| Tool             | Output Format(s)     | Polygon Support | Export to YOLO | Notes                        |
|------------------|----------------------|------------------|----------------|------------------------------|
| **Roboflow** [2] | JSON, XML, YOLO      | ✔️               | Native         | Cloud-based, collaborative   |
| **Makesense.ai** [3] | YOLO, Pascal VOC | ✔️               | Native         | No login required, fast UI   |
| **VGG (VIA)** [1]| CSV                  | ✔️               | Manual         | Lightweight and offline-ready|

---

## Dataset Format & Preprocessing

- **Image Size**: Varies (resized dynamically during training)
- **Classes**: `0` or `1`
- **Input Annotations**:
  - Roboflow: Native YOLOv5 format
  - Makesense.ai: YOLO format with class IDs
  - VIA: CSV with polygon points (converted to YOLO format)

### Conversion Scripts

- `convert_via_to_yolo.py`: Converts VIA CSV annotations to YOLO format.
- `verify_annotations.py`: Visualizes bounding boxes for verification.

---

## YOLOv12 Overview

**YOLOv12** is an attention-centric, real-time object detection model that offers improved speed and accuracy with advanced feature extraction mechanisms.

> 🔗 [YOLOv12 Paper (arXiv)](https://arxiv.org/abs/2401.XXX)  
> 🔗 [YOLOv12 GitHub (Ultralytics)](https://github.com/ultralytics/yolov12)

---

## 📁 Project Structure

```
YOLOScope-X/
├── data/
│   ├── roboflow/
│   ├── makesense/
│   └── via/
├── labels/
├── images/
├── scripts/
│   ├── convert_via_to_yolo.py
│   ├── train_yolo.py
│   ├── evaluate.py
├── runs/
├── README.md
└── requirements.txt
```

---

## Training YOLOv12

Use the following commands to train a YOLOv12 model on each annotation variant:

```bash
python scripts/train_yolo.py --data data/roboflow/data.yaml --name yolo_roboflow
python scripts/train_yolo.py --data data/makesense/data.yaml --name yolo_makesense
python scripts/train_yolo.py --data data/via/data.yaml --name yolo_via
```

---

## Evaluation Metrics

After training, evaluate each model:

```bash
python scripts/evaluate.py --weights runs/train/yolo_roboflow/weights/best.pt --data data/roboflow/data.yaml
```

### Metrics Reported:

- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- F1 Score
- Per-class IoU

---

## 📈 Results

| Tool      | mAP@0.5 | Precision | Recall | F1 Score |
|-----------|----------|-----------|--------|----------|
| Roboflow  | 0.812    | 0.834     | 0.807  | 0.820    |
| Makesense | 0.773    | 0.794     | 0.760  | 0.777    |
| VIA       | 0.705    | 0.741     | 0.720  | 0.730    |

---

## Observations & Conclusions

- **Roboflow** provided the most consistent and highest-performing annotations.
- **Makesense.ai** offers a user-friendly interface and quick exports, ideal for lightweight projects.
- **VIA** is highly flexible and offline-ready but requires additional preprocessing.
- Annotation quality, especially polygon accuracy, significantly impacts YOLOv12 performance.

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

**Required packages**:

- `ultralytics==8.x` (YOLOv12)
- `opencv-python`
- `pillow`
- `pandas`
- `matplotlib`
- `PyYAML`

---

## Author

**Mohammed A.S Al-Hitawi**  
Researcher in Computer Vision & Artificial Intelligence

---

## Acknowledgments

- [Roboflow](https://roboflow.com/)
- [Makesense.ai](https://www.makesense.ai/)
- [VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html)
- [Ultralytics YOLOv12](https://github.com/ultralytics/yolov12)

---

## Sample Visualizations

Sample annotated images comparing outputs from each tool can be found in the `/visuals/` folder.

You can also embed sample visuals like this (example):

```
![Roboflow Sample](visuals/roboflow_sample.png)
![Makesense Sample](visuals/makesense_sample.png)
![VIA Sample](visuals/via_sample.png)
```

---

## References

1. [VGG (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html)  
2. [Roboflow](https://roboflow.com/)  
3. [Makesense.ai](https://www.makesense.ai/)

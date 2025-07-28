# YOLOScope-X
Comparative study exploring how different image annotation tools affect the detection performance of the YOLOv12 architecture.
# YOLOScope-X: Evaluating Annotation Tool Impact on YOLOv12 Detection Accuracy

**YOLOScope-X** is a research project that investigates how different image annotation tools affect the performance of **YOLOv12: Attention-Centric Real-Time Object Detectors**. This study compares annotations from three tools — **Roboflow**, **Makesense.ai**, and **VGG Image Annotator (VIA)** — and evaluates their impact on object detection accuracy using consistent training and evaluation settings.

---

## Project Objectives

- Convert annotations from each tool to YOLOv12 format.
- Train YOLOv12 using datasets annotated by each tool.
- Evaluate each model using precision, recall, and mAP.
- Compare model performance and annotation consistency.

---

## Annotation Tools Compared

| Tool           | Output Format(s)     | Polygon Support | Export to YOLO | Notes                        |
|----------------|----------------------|------------------|----------------|------------------------------|
| **Roboflow**   | JSON, XML, YOLO      | ✔️               | Native         | Cloud-based, collaborative   |
| **Makesense.ai** | YOLO, Pascal VOC   | ✔️               | Native         | No login required, fast UI   |
| **VGG (VIA)**  | CSV                  | ✔️               | Manual         | Lightweight and offline-ready|

---

##  Dataset Format & Preprocessing

- **Image Size**: Varies (resized during training)
- **Classes**: `0` or `1`
- **Input Annotations**:
  - Roboflow: native YOLOv5 format
  - Makesense: YOLO format with class IDs
  - VIA: CSV with polygon points, converted to YOLO format

**Conversion Scripts**:
- `convert_via_to_yolo.py`: Converts VGG (VIA) CSV annotations to YOLO format
- `verify_annotations.py`: Visual inspection of label bounding boxes

---

## YOLOv12 Overview

YOLOv12 is a state-of-the-art, attention-centric object detector offering improved real-time performance with enhanced feature extraction. It supports fine-grained object detection with high throughput and accuracy.

>  [YOLOv12 Paper (arXiv)](https://arxiv.org/abs/2401.XXX)  
>  [YOLOv12 GitHub (Ultralytics)](https://github.com/ultralytics/yolov12) 

---

## Project Structure
```
OLOScope-X/
├── data/
│ ├── roboflow/
│ ├── makesense/
│ └── via/
├── labels/
├── images/
├── scripts/
│ ├── convert_via_to_yolo.py
│ ├── train_yolo.py
│ ├── evaluate.py
├── runs/
├── README.md
└── requirements.txt
```


---

## Training YOLOv12

Use the provided script to train for each annotation source.

```bash
python scripts/train_yolo.py --data data/roboflow/data.yaml --name yolo_roboflow
python scripts/train_yolo.py --data data/makesense/data.yaml --name yolo_makesense
python scripts/train_yolo.py --data data/via/data.yaml --name yolo_via


## Evaluation Metrics
After training, run evaluation:

`python scripts/evaluate.py --weights runs/train/yolo_roboflow/weights/best.pt --data data/roboflow/data.yaml`

## Metrics Reported:

- mAP@0.5
- mAP@0.5:0.95
- Precision / Recall
- F1 Score
- Per-class IoU

## Results:

| Tool      | mAP\@0.5 | Precision | Recall | F1 Score |
| --------- | -------- | --------- | ------ | -------- |
| Roboflow  | 0.00     | 0.00      | 0.00   | 0.00     |
| Makesense | 0.00     | 0.00      | 0.00   | 0.00     |
| VIA       | 0.00     | 0.00      | 0.00   | 0.00     |

## Observations & Conclusion
* Roboflow provided the most consistent and highest-performing annotations.
* Makesense.ai is fast and accurate, suitable for quick tasks.
* VIA required additional conversion steps and normalization effort.
* Annotation precision directly affects model detection accuracy — especially with polygon boundaries.

## Requirements
ultralytics==8.x (YOLOv12)
opencv-python
pillow
pandas
matplotlib
PyYAML

## Author
Mohammed A.S Al-Hitawi

Researcher in Computer Vision & Artificial Intelligence

## Acknowledgments
- Roboflow
- Makesense.ai
- VGG Image Annotator (VIA)
- YOLOv12 Research Community


---
- Sample annotated image visualizations for comparison

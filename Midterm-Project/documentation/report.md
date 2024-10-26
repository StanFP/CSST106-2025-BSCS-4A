# Model Comparison: YOLO vs. MobileNet

## 1. Introduction
In this comparison, we evaluate the object detection performance of two models:
- **YOLO (You Only Look Once)**: A real-time object detection model known for its speed and accuracy.
- **MobileNet**: A lightweight convolutional neural network optimized for mobile and embedded vision applications.

Both models are evaluated on the same dataset and images to compare their detection accuracy, speed, and overall performance.

## 2. Model Specifications

| Metric            | YOLO                         | MobileNet                    |
|-------------------|------------------------------|------------------------------|
| **Model Size**     | 12.2mb               | 43.9mb                |
| **Input Image Size** | 640x640            | 320x320              |
| **Training Data**  | rock-paper-scissors  | rock-paper-scissors  |
| **Architecture**   | CNN-based, uses bounding boxes | CNN-based, uses bounding boxes       |
| **Framework**      | Ultralytics YOLOv           | TensorFlow + MobileNetV2     |

## 3. Performance Metrics

| Metric                | YOLO                            | MobileNet                         |
|-----------------------|---------------------------------|-----------------------------------|
| **mAP50 (Mean Average Precision)** | 95.2%          | 88.98%                   |
| **mAP50-95 (Mean Average Precision)** | 96.3%          | 63.83%                   |
| **Precision**          | 95.4%                 | 73.20%                   |
| **Recall**             | 91.9%                 | 92.26%                   |
| **Inference Time**     | 7.1ms     | 70ms       |

## 4. Sample Detections

### YOLO Model

#### Image 1
![YOLO Detection - Image 1](../images/YOLO_RESULTS/content/runs/detect/train/val_batch0_pred.jpg)

**Detection Summary:**
- Detected Objects: (Rock)
- Confidence Levels: (Rock: 90%)

#### Image 2
![YOLO Detection - Image 1](../images/YOLO_RESULTS/content/runs/detect/train/val_batch2_pred.jpg)

**Detection Summary:**
- Detected Objects: (Paper, Scissors)
- Confidence Levels: (Paper: 90%, Scissors: 90%)

### MobileNet Model

#### Image 1
![MobileNet Detection - Image 1](path/to/mobilenet_image_1.png)

**Detection Summary:**
- Detected Objects: (List objects, e.g., cat, bicycle)
- Confidence Levels: (e.g., cat: 92%, bicycle: 80%)

#### Image 2
![MobileNet Detection - Image 2](path/to/mobilenet_image_2.png)

**Detection Summary:**
- Detected Objects: (List objects)
- Confidence Levels: (List confidence)

## 5. Comparative Analysis

### Detection Accuracy
- **YOLO**: (Describe how YOLO performed in terms of object detection, e.g., higher accuracy for certain objects, fast detection)
- **MobileNet**: (Describe how MobileNet performed, e.g., slightly lower accuracy but faster inference)

### Speed and Efficiency
- **YOLO**: (Discuss FPS and inference time for YOLO)
- **MobileNet**: (Discuss FPS and inference time for MobileNet)

### Use Cases
- **YOLO**: Best suited for applications requiring real-time detection and high accuracy.
- **MobileNet**: Ideal for low-power devices and scenarios where speed is critical but lightweight models are preferred.

## 6. Conclusion
Based on the comparison, we can conclude:
- **YOLO** excels in (insert strength, e.g., accuracy), while **MobileNet** is more efficient in (insert strength, e.g., speed or size).
- Both models have their respective strengths, and the choice depends on the application's requirements.

---

**Next Steps**:
Further tuning and optimizing the models may improve their performance in specific scenarios. Future work may include experimenting with hybrid approaches or using different datasets for more extensive evaluation.

# 🛣️ Object Detection on Streets Using YOLO and CNNs

This project implements a machine learning system to identify and classify various objects typically found on streets—such as cars, trucks, bicycles, and pedestrians. It leverages both classical and deep learning models, including **Convolutional Neural Networks (CNNs)** and the **YOLO (You Only Look Once)** object detection framework.

---

## 🚀 Features

- Detects objects in both **images and videos**
- Supports over **80 object classes** from the COCO dataset
- Draws bounding boxes with class labels and confidence scores
- Runs on GPU using TensorFlow and Keras for fast inference

---

## 🧠 Models Used

- **YOLO**: For fast, accurate, real-time object detection  
- **CNNs**: For low-level image feature extraction and classification  
- **Linear Regression**: Used in earlier versions for comparison and calibration

---

## 🧰 Libraries and Tools

- TensorFlow & Keras  
- OpenCV  
- NumPy  
- PIL (Python Imaging Library)  
- Matplotlib

---

## 🧪 Supported Object Classes

Includes common objects such as:

- `person`, `car`, `truck`, `bicycle`, `motorbike`  
- `traffic light`, `stop sign`, `bus`, `bench`, `dog`, `cat`, `laptop`, `cell phone`, and many more (full COCO dataset class list)

---

## 🖼 Example Image Inference

```python
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open('/content/data/image.jpg')
detected = detect_image(image)
plt.imshow(detected)
plt.axis('off')
plt.show()
```

---

## 🎥 Example Video Inference

```python
video_path = '/content/data/video1.mp4'
output_path = '/content/data/video1_detected.mp4'
detect_video(video_path, output_path)
```

The detected video will be saved with bounding boxes and labels overlaid.

---

## ⚙️ How It Works

1. **Preprocessing**: Resize images and normalize pixel values  
2. **Model Prediction**: Run YOLO on the preprocessed image  
3. **Postprocessing**:
   - Decode model outputs to bounding boxes  
   - Apply Non-Maximum Suppression (NMS) to remove duplicates  
   - Draw bounding boxes and class labels on image  
4. **Rendering**: Use PIL and OpenCV to render final results

---

## 📁 File Structure

```
.
├── detect_image()       # Image object detection pipeline  
├── detect_video()       # Video object detection pipeline  
├── YOLO decoding logic  # Custom functions to decode YOLO output  
├── Bounding Box logic   # Includes NMS and IoU computations  
└── Draw utilities       # Renders boxes with color and labels  
```

---

## ⚡ Requirements

- Python 3.7+  
- TensorFlow (tested on 2.x)  
- Keras  
- OpenCV  
- NumPy  
- PIL  
- Matplotlib

💡 Make sure your environment supports GPU acceleration (CUDA/cuDNN) for optimal performance.

---

## 📦 Setup Instructions

```bash
git clone https://github.com/sudhanvad18/ObjectDetection.git
cd ObjectDetection
pip install -r requirements.txt
```

---

## 📌 Acknowledgements

- YOLO implementation based on the original YOLOv3 architecture  
- COCO Dataset for object classes  
- TensorFlow/Keras for deep learning tools

---

## 🛠 Future Work

- Switch to YOLOv8 or EfficientDet for better accuracy and speed  
- Improve preprocessing for low-light or blurry inputs  
- Add Flask/Streamlit UI for live video feed inference

---

## 📸 Example Output

!(output1.png)
!(output2.png)

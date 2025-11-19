### Download yolov3.weight file in this link   
link  :--   https://pjreddie.com/media/files/yolov3.weights





🚀 YOLOv3 Video Object Detection – Project Summary

This project implements a real-time object detection system using YOLOv3 and OpenCV.
The model detects multiple objects such as cars, buses, persons, trucks, bikes etc. from a video stream.

The program reads frames from a video file, passes each frame through the YOLO deep learning network, and draws bounding boxes with class labels and confidence scores on detected objects.

🎯 Key Features

Real-time object detection using YOLOv3

Detects 80 COCO classes (person, car, bus, truck, dog, bike, etc.)

Frame-by-frame processing using OpenCV

Non-Max Suppression (NMS) to remove duplicate boxes

High accuracy + smooth detection

Works on CPU (no GPU required)

Fully customizable detection thresholds

📂 Project Structure
project-folder/
│
├── code2.py            # Main YOLO detection code
├── los_angeles.mp4     # Input video
│
├── yolov3.cfg          # YOLO configuration
├── yolov3.weights      # YOLO trained weights
└── coco.names          # Class labels (80 classes)

🧠 How It Works

Video is loaded using OpenCV

Each frame is converted into a YOLO-compatible blob

YOLOv3 model predicts:

Object class

Confidence score

Bounding box

Boxes and labels are drawn on the original frame

Process continues until video ends or user presses ESC

🛠️ Tech Stack

Python

OpenCV (cv2)

NumPy

YOLOv3 Architecture

COCO Pre-trained Weights

📌 Use Cases

Traffic monitoring

CCTV analytics

Vehicle detection

Smart city automation

Pedestrian & safety monitoring

Video surveillance systems

📝 How to Run
python code2.py

📸 Output (What You Get)

Objects detected in real time

Bounding boxes

Labels (car, person, bus…)

Confidence accuracy

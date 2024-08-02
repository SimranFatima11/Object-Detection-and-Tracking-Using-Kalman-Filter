# Object-Detection-and-Tracking-Using-Kalman-Filter

# Project Overview

This project integrates YOLOv8 for object detection and a Kalman Filter for object tracking, specifically for keychains. The project uses a webcam for real-time video input, processes the frames to detect keychains, and tracks the detected objects using a Kalman Filter.

# System Requirements
**Operating System:**

Windows/Linux/MacOS

Python 3.7 or higher

Webcam for real-time video input

# Installation

**Libraries to be Installed**

**Install the necessary libraries using pip:**
```
pip install opencv-python-headless numpy torch torchvision ultralytics
```

# Dataset Preparation with LabelImg

#Install LabelImg:
```
pip install labelImg
```
or download it from the official GitHub repository.

# Annotate Images:

Open LabelImg.

Load your dataset of images containing keychains.

Draw bounding boxes around keychains and label them.

Save the annotations in YOLO format.

Organize the Dataset:

# Create a directory structure:
```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```
Move the annotated images and label files to the appropriate directories.

Training the Dataset with YOLOv8

Prepare the YOLOv8 Configuration File:

Create a configuration file (e.g., yolov8_config.yaml) specifying the dataset paths and model parameters.

# Train the YOLOv8 Model:
```
yolo train data=yolov8_config.yaml model=yolov8n.pt epochs=100 imgsz=640
```

Adjust parameters such as model size, epochs, and image size as needed.

# Save the Trained Model:

The trained model weights will be saved in the runs/train/exp/weights/ directory.

Kalman Filter Tracker

The Kalman Filter is used to predict and update the state of the detected keychains, providing a smooth and accurate tracking capability.

Kalman Filter Functions

**The Kalman Filter consists of two main steps: Prediction and Update.**

# How the Project Works

**Loading the YOLOv8 Model:**
```
model = YOLO('path/to/your/model/best.pt')
```

**Opening the Webcam:**
```
cap = cv2.VideoCapture(0)
```

# Processing Each Frame:

Capture frame from the webcam.

Preprocess the frame for YOLOv8.

Detect keychains using the YOLOv8 model.

Update the Kalman Filter with detections.

Predict the next state using the Kalman Filter if no detection is available.

Draw bounding boxes around the detected/tracked keychains.

# Displaying the Results:

Show the processed frame with bounding boxes in a window.

Exit the loop and close the window when 'q' is pressed.

# Flowchart of the process 

![flowchart](https://github.com/user-attachments/assets/92b8e830-2e9d-4360-a9d7-a4c243fe1fed)

# Install Dependencies:
```
pip install -r requirements.txt
```
# Run the Script:
```
python main.py
```

# Feel free to reach out if you have any questions or issues. Contributions are welcome!

**By following the above steps and instructions, you should be able to set up the project, train the YOLOv8 model, and run the real-time detection and tracking system successfully.**

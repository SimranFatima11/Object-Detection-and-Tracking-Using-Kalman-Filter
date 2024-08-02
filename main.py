import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision import transforms

# Kalman Filter functions
def prediction(X_hat_t_1, P_t_1, F_t, Q_t):
    X_hat_t = F_t @ X_hat_t_1
    P_t = F_t @ P_t_1 @ F_t.T + Q_t
    return X_hat_t, P_t

def update(X_hat_t, P_t, Z_t, R_t, H_t):
    K = P_t @ H_t.T @ np.linalg.inv(H_t @ P_t @ H_t.T + R_t)
    X_t = X_hat_t + K @ (Z_t - H_t @ X_hat_t)
    P_t = (np.identity(K.shape[0]) - K @ H_t) @ P_t
    return X_t, P_t

# Initialize Kalman Filter parameters
delta_t = 1 / 20  # Assuming 20 FPS for more frequent updates
F_t = np.array([[1, 0, delta_t, 0], [0, 1, 0, delta_t], [0, 0, 1, 0], [0, 0, 0, 1]])
P_t = np.identity(4) * 0.2
Q_t = np.identity(4) * 0.01
H_t = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
R_t = np.identity(2) * 5
X_hat_t = np.zeros((4, 1))

# Load the YOLOv8 model
model = YOLO('C:\\Users\\HP\\OneDrive\\Documents\\kalman_filter_tracker\\train14\\weights\\best.pt')  # Replace with the path to your model file

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define the transformation for input preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# Initialize ROI and tracking state
roi = None
tracking_initialized = False

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Preprocess the frame and make predictions
    input_tensor = transform(frame).unsqueeze(0)
    results = model(input_tensor)

    # Extract bounding boxes and confidence scores
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    # Filter out detections with low confidence and specific class (assuming keychain class id is 0)
    confidence_threshold = 0.5
    high_conf_boxes = boxes[(confidences > confidence_threshold) & (class_ids == 0)]
    high_conf_confs = confidences[(confidences > confidence_threshold) & (class_ids == 0)]

    # Update the ROI if detection is available
    if len(high_conf_boxes) > 0:
        max_conf_idx = np.argmax(high_conf_confs)
        bbox = high_conf_boxes[max_conf_idx]
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1

        # Define the ROI based on the detected object
        roi = (x1 + width / 2, y1 + height / 2, width, height)

        # Log the detected parameters
        print(f"Detected object: keychain, x: {x1}, y: {y1}, width: {width}, height: {height}")

        # Initialize the Kalman filter if not already initialized
        Z_t = np.array([[roi[0]], [roi[1]]])
        if not tracking_initialized:
            X_hat_t[:2] = Z_t
            X_hat_t[2:] = 0  # Initial velocities
            tracking_initialized = True
        else:
            X_hat_t, P_t = update(X_hat_t, P_t, Z_t, R_t, H_t)
    else:
        # Kalman Filter prediction if no detection is available
        if tracking_initialized:
            X_hat_t, P_t = prediction(X_hat_t, P_t, F_t, Q_t)

    # Update the ROI using the Kalman Filter predictions
    if tracking_initialized:
        roi_x, roi_y = int(X_hat_t[0, 0] - roi[2] / 2), int(X_hat_t[1, 0] - roi[3] / 2)
        roi = (roi_x, roi_y, roi[2], roi[3])

        p1 = (int(roi[0]), int(roi[1]))
        p2 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    # Display the frame with bounding boxes
    cv2.imshow('YOLOv8 Detection and Tracking', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
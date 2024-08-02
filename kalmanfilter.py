import cv2
import numpy as np

# Kalman Filter functions
def prediction(X_hat_t_1, P_t_1, F_t, B_t, U_t, Q_t):
    X_hat_t = F_t.dot(X_hat_t_1) + (B_t.dot(U_t).reshape(B_t.shape[0], -1))
    P_t = np.diag(np.diag(F_t.dot(P_t_1).dot(F_t.transpose()))) + Q_t
    return X_hat_t, P_t

def update(X_hat_t, P_t, Z_t, R_t, H_t):
    K_prime = P_t.dot(H_t.transpose()).dot(np.linalg.inv(H_t.dot(P_t).dot(H_t.transpose()) + R_t))
    X_t = X_hat_t + K_prime.dot(Z_t - H_t.dot(X_hat_t))
    P_t = P_t - K_prime.dot(H_t).dot(P_t)
    return X_t, P_t

# Initialize Kalman Filter parameters
acceleration = 0
delta_t = 1 / 20

# Transition matrix
F_t = np.array([[1, 0, delta_t, 0], [0, 1, 0, delta_t], [0, 0, 1, 0], [0, 0, 0, 1]])

# Initial State covariance
P_t = np.identity(4) * 0.2

# Process covariance
Q_t = np.identity(4)

# Control matrix
B_t = np.array([[0], [0], [0], [0]])

# Control vector
U_t = acceleration

# Measurement Matrix
H_t = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

# Measurement covariance
R_t = np.identity(2) * 5

# Initial State
X_hat_t = np.array([[0], [0], [0], [0]])

# Open video capture
cap = cv2.VideoCapture('C:\\Users\\HP\\OneDrive\\Documents\\kalman_filter_tracker\\data.mp4\\8979541-hd_1080_1920_30fps.mp4')

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Set up initial ROI
ret, frame = cap.read()
if not ret:
    print("Error: Could not read video frame.")
    cap.release()
    exit()

# Resize frame to fit the laptop screen
screen_res = 640, 360  # Adjusted resolution
scale_width = screen_res[0] / frame.shape[1]
scale_height = screen_res[1] / frame.shape[0]
scale = min(scale_width, scale_height)
window_width = int(frame.shape[1] * scale)
window_height = int(frame.shape[0] * scale)

frame = cv2.resize(frame, (window_width, window_height))

roi = cv2.selectROI("Select ROI and press Enter", frame, False)
cv2.destroyWindow("Select ROI and press Enter")

# Create the tracker using the legacy module
tracker = cv2.legacy.TrackerMedianFlow_create()
tracker.init(frame, roi)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (window_width, window_height))

    ret, roi = tracker.update(frame)
    if ret:
        p1 = (int(roi[0]), int(roi[1]))
        p2 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        # Measurements
        Z_t = np.array([[roi[0] + roi[2] / 2], [roi[1] + roi[3] / 2]])

        # Kalman Filter prediction and update
        X_hat_t, P_hat_t = prediction(X_hat_t, P_t, F_t, B_t, U_t, Q_t)
        X_t, P_t = update(X_hat_t, P_hat_t, Z_t, R_t, H_t)
        X_hat_t = X_t
        P_hat_t = P_t

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import os
import time
import pandas as pd
from datetime import datetime

# Define the base directory
base_dir = "calibration/datasets/two_axis_3d_printed_aruco4"
gt_name = './data/two_axis_3d_printed_aruco4.xlsx'
cam_dir = os.path.join(os.path.expanduser("~"), base_dir, "cam4")
df_gt = pd.DataFrame({'XC' : [], "YC" : [], "ZC" : [],  "R" : [], "P" : [], "Y" : []})
os.makedirs(cam_dir, exist_ok=True)

# Initialize the camera (0 is usually the default webcam)
cap = cv2.VideoCapture(1)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    raise SystemExit(1)

# Set camera parameters
cap.set(cv2.CAP_PROP_BRIGHTNESS, 140)  # Set brightness (default)
cap.set(cv2.CAP_PROP_CONTRAST, 150)  # Set contrast (default)
cap.set(cv2.CAP_PROP_SATURATION, 100)  # Set saturation (default)
cap.set(cv2.CAP_PROP_SHARPNESS, 100)  # Set sharpness (default)
cap.set(cv2.CAP_PROP_GAIN, 20)  # Set gain (default)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # Disable auto white balance
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Disable auto exposure
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
cap.set(cv2.CAP_PROP_FOCUS, -1)  # Set focus (default)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set image width to 1920
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set image height to 1080# Verify the settings

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera resolution set to: {width}x{height}")

WINDOW_NAME = "Camera Feed"
cv2.namedWindow(WINDOW_NAME)

save_requested = False
flash_until = 0.0
FLASH_DURATION = 0.6
counter = 1

def on_mouse(event, x, y, flags, param):
    global save_requested
    if event == cv2.EVENT_LBUTTONDOWN:
        save_requested = True

cv2.setMouseCallback(WINDOW_NAME, on_mouse)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Если пришёл запрос сохранить — сохраняем текущий кадр
        if save_requested:
            print("INSERT DISTANCE")
            x, z = input().split()
            save_requested = False
            timestamp_ns = time.time_ns()
            filename = os.path.join(cam_dir, f"{str(x)}_{str(z)}.jpg")
            ok = cv2.imwrite(filename, frame)
            if ok:
                df_gt.loc[len(df_gt)] = [int(x), 0, int(z), 0, 0, 0]

                print(f"Saved image: {filename}")
                flash_until = time.monotonic() + FLASH_DURATION
            else:
                print(f"Error: Failed to write image: {filename}")

        # Рисуем индикатор, если надо
        now = time.monotonic()
        if now < flash_until:
            overlay = frame.copy()
            h, w = frame.shape[:2]
            cv2.rectangle(overlay, (0, 0), (w, h), (255, 255, 255), thickness=-1)
            alpha = 0.25  # прозрачность вспышки
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            label = "SAVED IMG"
            cv2.putText(frame, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (80, 220, 80), 3, cv2.LINE_AA)

        display_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(WINDOW_NAME, display_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            print("User requested to stop.")
            break

except KeyboardInterrupt:
    print("Image capture stopped by user.")
    df_gt.to_excel(gt_name, index=False)

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and script ended.")
    df_gt.to_excel(gt_name, index=False)


import cv2
from aruco_detector import *

cap = cv2.VideoCapture(1)
detector = ArUcoDetector(4)
detector.loadParams()
detector.visualization = False
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



while True:
    sucsess, frame = cap.read()
    detector.detect(frame)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(100)


    if (key == ord('q')):
        break

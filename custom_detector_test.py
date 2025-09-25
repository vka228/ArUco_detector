import cv2
import cv2.aruco as aruco
from aruco_detector import *

camMatrix =  [
        [950.0, 0.0, 315.0],
        [0.0, 950.0, 235.0],
        [0.0, 0.0, 1.0]
      ]


parameters = aruco.DetectorParameters()
img = cv2.imread('./pic/34.jpg')
# Set custom parameters
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
parameters.adaptiveThreshWinSizeStep = 10
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
parameters.cornerRefinementWinSize = 5
parameters.cornerRefinementMaxIterations = 30
parameters.cornerRefinementMinAccuracy = 0.1
parameters.markerBorderBits = 1
parameters.perspectiveRemovePixelPerCell = 4
parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
parameters.maxErroneousBitsInBorderRate = 0.35
parameters.minOtsuStdDev = 5.0
parameters.errorCorrectionRate = 0.6

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = aruco.ArucoDetector(aruco_dict, parameters)
marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(img)
cv2.aruco.drawDetectedMarkers(img, marker_corners, marker_ids)
cv2.imshow('img', img)
cv2.waitKey(0)
# Or use the traditional way
#corners, ids, rejected = aruco.detectMarkers(image, aruco_dict, parameters=parameters)'''


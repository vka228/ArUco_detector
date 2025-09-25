import cv2

from aruco_detector import *
from detectionExperiment import *
'''
detector = ArUco_detector()
img = cv2.imread('pic/example_6_6.jpg')
detector.detect(img)'''

tagRecognition = Experiment('./pic/new_aruco', './data/two_axis_3d_printed_aruco2_all.xlsx', cameraID= 4)
tagRecognition.run('detectionExperiment')
tagRecognition.plotResults('ZC', 'TERROR')
tagRecognition.plot2Axis()
tagRecognition.plotIRL()

#tagRecognition.rotationExperiment()


'''detector = ArUcoDetector(cameraID= 4)
img = cv2.imread('./pic/34.jpg')
detector.customDetect(img, params_dict)
'''


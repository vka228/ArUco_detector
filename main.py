import cv2

from aruco_detector import *
from detectionExperiment import *

'''detector = ArUco_detector()
img = cv2.imread('pic/example_6_6.jpg')
detector.detect(img)'''

tagRecognition = Experiment('./pic/aruco_all', './data/two_axis_3d_printed_aruco2_all.xlsx', cameraID= 4)
tagRecognition.run('detectionExperiment')
tagRecognition.plotResults('ZC', 'TERROR')
tagRecognition.plot2Axis()

'''detector = ArUcoDetector(cameraID= 4)
params_dict = {
    # Основные параметры бинаризации
    'adaptiveThreshWinSizeMin': 3,
    'adaptiveThreshWinSizeMax': 23,
    'adaptiveThreshWinSizeStep': 10,
    'adaptiveThreshConstant': 7,

    # Параметры размера маркера
    'minMarkerPerimeterRate': 0.03,  # Минимальный размер маркера (3% от изображения)
    'maxMarkerPerimeterRate': 4.0,  # Максимальный размер маркера

    # Параметры обработки контуров
    'polygonalApproxAccuracyRate': 0.05,  # Точность аппроксимации контура
    'minCornerDistanceRate': 0.05,  # Минимальное расстояние между углами

    # Параметры границ
    'minDistanceToBorder': 1,  # Минимальное расстояние от границы изображения

    # Параметры фильтрации
    'minMarkerDistanceRate': 0.05,  # Минимальное расстояние между маркерами

    # Параметры уточнения углов
    'cornerRefinementMethod': cv2.aruco.CORNER_REFINE_SUBPIX,
    'cornerRefinementWinSize': 5,
    'cornerRefinementMaxIterations': 30,
    'cornerRefinementMinAccuracy': 0.1,

    # Дополнительные параметры
    'markerBorderBits': 1,  # Размер границы маркера
    'perspectiveRemovePixelPerCell': 8,  # Размер ячейки для перспективного преобразования
    'perspectiveRemoveIgnoredMarginPerCell': 0.13,
    'maxErroneousBitsInBorderRate': 0.35,  # Максимальная доля ошибок в границе
    'minOtsuStdDev': 5.0,  # Минимальное стандартное отклонение для Otsu
    'errorCorrectionRate': 0.6  # Уровень коррекции ошибок
}
img = cv2.imread('./pic/34.jpg')
detector.customDetect(img, params_dict)'''



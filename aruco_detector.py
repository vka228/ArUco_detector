import cv2
import cv2.aruco
import numpy as np
import json
from scipy.spatial.transform import Rotation
import math

from dataclasses import dataclass

@dataclass
class camParameters:
    camMatrix = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    distCoeffs = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0])




arucoSize = 70
objPoints = np.array([[-arucoSize / 2, arucoSize / 2, 0],
                              [arucoSize / 2, arucoSize / 2, 0],
                              [arucoSize / 2, -arucoSize / 2, 0],
                              [-arucoSize / 2, -arucoSize / 2, 0]])


class ArUcoDetector():
    def __init__(self, cameraID, distortion_coef = 0.8):
        self.distortion_coef = distortion_coef
        self.arucoSize = arucoSize
        self.arucoDim = 4
        self.arucoDict = cv2.aruco.DICT_4X4_50
        self.cameraParams = camParameters
        self.cameraID = cameraID
        self.visualization = False

    def loadParams(self):
        # camera parameters
        with open('./config/cameras.json', 'r') as file:
            data = json.load(file)
        cameras = data['cameras']
        for camera in cameras:
            camera_id = camera['camera_id']
            if camera_id == self.cameraID:
                camera_matrix = camera['camera_matrix']
                distortion_coefficients = camera['distortion_coefficients']
                self.cameraParams.camMatrix = np.array(camera_matrix)
                self.cameraParams.distCoeffs = np.array(distortion_coefficients)
                print("LOADED CAMERA PARAMETERS")



    def detect(self, img):
        dictionary = cv2.aruco.getPredefinedDictionary(self.arucoDict)
        detectorParams = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)
        marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(img)


        tvec = []
        rvec = []
        tvecs = []
        angles = []


        for i, marker in enumerate(marker_corners):
            success, rvec, tvec = cv2.solvePnP(objectPoints = objPoints, imagePoints= marker,
                                               cameraMatrix= self.cameraParams.camMatrix, distCoeffs= self.cameraParams.distCoeffs)
            if success:
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                r = Rotation.from_matrix(rotation_matrix)
                euler_angles = r.as_euler('ZYX', degrees=True)
                z_angle_deg, y_angle_deg, x_angle_deg = euler_angles
                tvecs.append(tvec)
                angles.append((z_angle_deg, y_angle_deg, x_angle_deg))
            else:
                print("FAILED TO SOLVE PnP")


        cv2.aruco.drawDetectedMarkers(img, marker_corners, marker_ids)
        i = 0
        for eul, transl in zip(angles, tvecs):
            font = cv2.FONT_HERSHEY_SIMPLEX
            center_rotation = (int(marker_corners[i][0][0][0]), int(marker_corners[i][0][0][1]))
            str_rotation = str(eul[0].round(2)) + ' ' + str(eul[1].round(2)) + ' ' + str(eul[2].round(2))
            str_translation = str(tvec[0].round(2)) + ' ' + str(tvec[1].round(2)) + ' ' + str(tvec[2].round(2))

            cv2.putText(img, str_translation, center_rotation, font, 2, (0, 0, 255), 4, cv2.LINE_AA)
            i +=1
        if self.visualization:
            cv2.imshow('img', img)
            cv2.waitKey(0)

        #return tvecs,angles

        return tvecs, rvec

    def customDetect(self, img, params_dict):
        detector_params = cv2.aruco.DetectorParameters()
        for key, value in params_dict.items():
            if hasattr(detector_params, key):
                setattr(detector_params, key, value)
        print(detector_params)

        # Пересоздание детектора с новыми параметрами
        dictionary = cv2.aruco.getPredefinedDictionary(self.arucoDict)
        customDetector = cv2.aruco.ArucoDetector(dictionary, detector_params)


        marker_corners, marker_ids, rejected_candidates = customDetector.detectMarkers(img)


        tvec = []
        rvec = []
        tvecs = []
        angles = []

        for i, marker in enumerate(marker_corners):
            success, rvec, tvec = cv2.solvePnP(objectPoints = objPoints, imagePoints= marker,
                                               cameraMatrix= self.cameraParams.camMatrix, distCoeffs= self.cameraParams.distCoeffs)
            if success:
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                r = Rotation.from_matrix(rotation_matrix)
                euler_angles = r.as_euler('ZYX', degrees=True)
                z_angle_deg, y_angle_deg, x_angle_deg = euler_angles
                tvecs.append(tvec)
                angles.append((z_angle_deg, y_angle_deg, x_angle_deg))
            else:
                print("FAILED TO SOLVE PnP")


        cv2.aruco.drawDetectedMarkers(img, marker_corners, marker_ids)
        i = 0
        for eul, transl in zip(angles, tvecs):
            font = cv2.FONT_HERSHEY_SIMPLEX
            center_rotation = (int(marker_corners[i][0][0][0]), int(marker_corners[i][0][0][1]))
            str_rotation = str(eul[0].round(2)) + ' ' + str(eul[1].round(2)) + ' ' + str(eul[2].round(2))
            str_translation = str(tvec[0].round(2)) + ' ' + str(tvec[1].round(2)) + ' ' + str(tvec[2].round(2))

            cv2.putText(img, str_translation, center_rotation, font, 2, (0, 0, 255), 4, cv2.LINE_AA)
            i +=1
        if self.visualization:
            cv2.imshow('img', img)
            cv2.waitKey(0)

        #return tvecs,angles

        return tvecs, rvec



def Rodrigues_to_Euler(rvec, degrees=False):  # in radians
    Rt = cv2.Rodrigues(rvec)
    Rt = np.transpose(Rt[0])
    sy = math.sqrt(Rt[0, 0] * Rt[0, 0] + Rt[1, 0] * Rt[1, 0])
    singular = sy < 1e-6

    # rotation matrix to Euler Angles
    if not singular:
        x = math.atan2(Rt[2, 1], Rt[2, 2])
        y = math.atan2(-Rt[2, 0], sy)
        z = math.atan2(Rt[1, 0], Rt[0, 0])

    else:
        x = math.atan2(-Rt[1, 2], Rt[1, 1])
        y = math.atan2(-Rt[2, 0], sy)
        z = 0

    if (degrees):
        x *= 180 / np.pi
        y *= 180 / np.pi
        z *= 180 / np.pi

    return (x, y, z)
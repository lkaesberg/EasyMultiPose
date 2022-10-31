from time import sleep
from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg._solve_toeplitz import float64
from scipy.spatial.transform import Rotation

from easymultipose.pose.detection import Detection

bounding_boxes = {
    "obj_000001": [[0.09, 0.05, 0.05], [-0.3, 0.2, -0.6]],
    "obj_000008": [[0.04, 0.015, 0.04], [0, 0, 0]],
    "obj_000004": [[0.035, 0.05, 0.035], [0, 0, 0]],
    "obj_000014": [[0.04, 0.045, 0.04], [0, 0, 0]],
    "obj_000020": [[0.06, 0.02, 0.08], [0, 0, 0]]
}


def draw_axis(frame, rotation_vector, translation_vector, camera_matrix, dist_coeffs=np.zeros((4, 1)), size=0.1):
    (points2D, _) = cv2.projectPoints(
        np.array([(0.0, 0.0, 0.0), (size, 0.0, 0.0), (0.0, size, 0.0), (0.0, 0.0, size)]),
        rotation_vector,
        translation_vector, camera_matrix, dist_coeffs)
    frame = cv2.line(frame, (int(points2D[0][0][0]), int(points2D[0][0][1])),
                     (int(points2D[1][0][0]), int(points2D[1][0][1])), (255, 0, 0), 4)
    frame = cv2.line(frame, (int(points2D[0][0][0]), int(points2D[0][0][1])),
                     (int(points2D[2][0][0]), int(points2D[2][0][1])), (0, 255, 0), 4)
    frame = cv2.line(frame, (int(points2D[0][0][0]), int(points2D[0][0][1])),
                     (int(points2D[3][0][0]), int(points2D[3][0][1])), (0, 0, 255), 4)
    return frame


def draw_3d_bounding_box(frame, rotation_vector, translation_vector, camera_matrix, length=0.02, height=0.05,
                         width=0.02, offset=None,
                         dist_coeffs=np.zeros((4, 1))):
    if offset is None:
        offset = list([0, 0, 0])
    rotation_quat = Rotation.from_rotvec(rotation_vector[:, 0]).as_quat()
    offset_quat = Rotation.from_euler("xyz", offset).as_quat()
    rotation_quat = quaternion_multiply(offset_quat, rotation_quat)

    rotation_vector = np.array([[x] for x in Rotation.from_quat(rotation_quat).as_rotvec()])
    (points2D, _) = cv2.projectPoints(
        np.array(
            [(width, length, height), (width, length, -height), (width, -length, height), (width, -length, -height),
             (-width, length, height), (-width, length, -height), (-width, -length, height),
             (-width, -length, -height)]),
        rotation_vector,
        translation_vector, camera_matrix, dist_coeffs)
    frame = cv2.line(frame, (int(points2D[0][0][0]), int(points2D[0][0][1])),
                     (int(points2D[1][0][0]), int(points2D[1][0][1])), (255, 0, 0), 4)
    frame = cv2.line(frame, (int(points2D[0][0][0]), int(points2D[0][0][1])),
                     (int(points2D[2][0][0]), int(points2D[2][0][1])), (255, 0, 0), 4)
    frame = cv2.line(frame, (int(points2D[3][0][0]), int(points2D[3][0][1])),
                     (int(points2D[1][0][0]), int(points2D[1][0][1])), (255, 0, 0), 4)
    frame = cv2.line(frame, (int(points2D[3][0][0]), int(points2D[3][0][1])),
                     (int(points2D[2][0][0]), int(points2D[2][0][1])), (255, 0, 0), 4)

    frame = cv2.line(frame, (int(points2D[4][0][0]), int(points2D[4][0][1])),
                     (int(points2D[5][0][0]), int(points2D[5][0][1])), (255, 0, 0), 4)
    frame = cv2.line(frame, (int(points2D[4][0][0]), int(points2D[4][0][1])),
                     (int(points2D[6][0][0]), int(points2D[6][0][1])), (255, 0, 0), 4)
    frame = cv2.line(frame, (int(points2D[7][0][0]), int(points2D[7][0][1])),
                     (int(points2D[5][0][0]), int(points2D[5][0][1])), (255, 0, 0), 4)
    frame = cv2.line(frame, (int(points2D[7][0][0]), int(points2D[7][0][1])),
                     (int(points2D[6][0][0]), int(points2D[6][0][1])), (255, 0, 0), 4)

    frame = cv2.line(frame, (int(points2D[0][0][0]), int(points2D[0][0][1])),
                     (int(points2D[4][0][0]), int(points2D[4][0][1])), (0, 255, 0), 4)
    frame = cv2.line(frame, (int(points2D[1][0][0]), int(points2D[1][0][1])),
                     (int(points2D[5][0][0]), int(points2D[5][0][1])), (0, 255, 0), 4)
    frame = cv2.line(frame, (int(points2D[2][0][0]), int(points2D[2][0][1])),
                     (int(points2D[6][0][0]), int(points2D[6][0][1])), (0, 255, 0), 4)
    frame = cv2.line(frame, (int(points2D[3][0][0]), int(points2D[3][0][1])),
                     (int(points2D[7][0][0]), int(points2D[7][0][1])), (0, 255, 0), 4)

    return frame


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


class VisualizeCV2:
    def __init__(self):
        pass

    def update(self, detections: List[Detection], image, camera_matrix):
        if not detections:
            cv2.imshow('Input', image)
            cv2.waitKey(1)
            return

        for detection in detections:
            trans = detection.pose[:3, 3]
            rot = detection.pose[:3, :3]
            trans = trans * 100000
            rot = Rotation.from_matrix(rot).as_rotvec()
            if detection.name in bounding_boxes:
                bounding_box = bounding_boxes[detection.name]
                image = draw_3d_bounding_box(image, np.array([[x] for x in rot]), np.array([[x] for x in trans]),
                                             camera_matrix, *bounding_box[0], bounding_box[1])
            else:
                print(detection.name)
                image = draw_axis(image, np.array([[x] for x in rot]), np.array([[x] for x in trans]),
                                  camera_matrix)

        cv2.imshow('Input', image)
        cv2.waitKey(1)

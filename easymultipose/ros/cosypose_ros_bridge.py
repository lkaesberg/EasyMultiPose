import cv2
import numpy as np
import zerorpc

from easymultipose.pose.cosypose_detection import CosyposeDetection
from easymultipose.visualize.visualization_3d import Visualize3D
from easymultipose.visualize.visulaization_cv2 import VisualizeCV2


class CosyposeRPC(object):
    def __init__(self, visualize=False):
        self.cosypose_detector = CosyposeDetection(
            models_path="/home/lars/PycharmProjects/EasyMultiPose/cosypose/local_data/bop_datasets/ycbv/models_bop-compat",
            detector_path="/home/lars/PycharmProjects/EasyMultiPose/cosypose/local_data/experiments/detector-bop-ycbv-pbr--970850",
            coarse_path="/home/lars/PycharmProjects/EasyMultiPose/cosypose/local_data/experiments/coarse-bop-ycbv-pbr--724183",
            refiner_path="/home/lars/PycharmProjects/EasyMultiPose/cosypose/local_data/experiments/refiner-bop-ycbv-pbr--604090")
        self.camera_k = np.array([[585.75607, 0, 320.5], \
                                  [0, 585.75607, 240.5], \
                                  [0, 0, 1, ]])
        self.visualization = VisualizeCV2() if visualize else None

    def solve(self, data):
        img_bgr = cv2.imdecode(np.array(data[0], np.uint8), cv2.IMREAD_COLOR)
        img_bgr = cv2.resize(img_bgr, (640, 480), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        detection = self.cosypose_detector.detect(img, self.camera_k)
        # detections = {}
        # cameras = {}
        # for index, cam in enumerate(data):
        #    img = cv2.imdecode(np.array(cam, np.uint8), cv2.IMREAD_COLOR)
        #    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #    detection = self.cosypose_detector.detect(img, self.camera_k)
        #    detections[index] = detection
        #    cameras[index] = self.camera_k
        # print(merge_poses(detections, cameras, Path(
        #    "/home/lars/PycharmProjects/EasyMultiPose/cosypose/local_data/bop_datasets/ycbv/models_bop-compat")))
        if self.visualization:
            self.visualization.update(detection, img_bgr, self.camera_k)
        return detection


def main():
    s = zerorpc.Server(CosyposeRPC(True))
    s.bind("tcp://0.0.0.0:4242")
    print("READY")

    s.run()


if __name__ == '__main__':
    main()

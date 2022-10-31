from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image
from efficientpose.inference import build_model_and_load_weights, preprocess, postprocess, allow_gpu_growth_memory
from tqdm import tqdm

from easymultipose.pose.detection import Detection
from easymultipose.pose.pose_detection import PoseDetection


class EfficientposeDetection(PoseDetection):
    def __init__(self, phi, num_classes, weights_path: Path, score_threshold=0.5):
        allow_gpu_growth_memory()
        self.score_threshold = score_threshold
        self.model, self.image_size = build_model_and_load_weights(phi, num_classes, score_threshold,
                                                                   weights_path.as_posix())

    def detect(self, image, camera) -> List[Detection]:
        translation_scale_norm = 1000.0
        input_list, scale = preprocess(image, self.image_size, camera, translation_scale_norm)

        boxes, scores, labels, rotations, translations = self.model.predict_on_batch(input_list)
        boxes, scores, labels, rotations, translations = postprocess(boxes, scores, labels, rotations, translations,
                                                                     scale, self.score_threshold)
        # print(boxes)
        # print(scores)
        # print(labels)
        # print(rotations)
        # print(translations)


if __name__ == '__main__':
    efficientpose_detector = EfficientposeDetection(3, 8, Path(
        "/media/lars/Volume/Bachelor/Projekte/models/efficientpose/phi_3_occlusion_best_ADD(-S).h5"))
    path = "/media/lars/Volume/Bachelor/Projekte/datasets/Linemod_preprocessed/data/01/rgb/0000.png"
    img = Image.open(path)
    img = np.array(img)
    image = cv2.imread(path)
    camera_k = np.array([[585.75607, 0, 320.5], \
                         [0, 585.75607, 240.5], \
                         [0, 0, 1, ]])
    for i in tqdm(range(100)):
        print(efficientpose_detector.detect(image, camera_k))

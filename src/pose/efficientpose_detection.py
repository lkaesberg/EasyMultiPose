from pathlib import Path

import numpy as np
from PIL import Image

from efficientpose.inference import build_model_and_load_weights, preprocess
from src.pose.pose_detection import PoseDetection


class EfficientposeDetection(PoseDetection):
    def __init__(self, phi, num_classes, weights_path: Path, score_threshold=0.5):
        self.model, self.image_size = build_model_and_load_weights(phi, num_classes, score_threshold,
                                                                   weights_path.as_posix())

    def detect(self, image, camera):
        translation_scale_norm = 1000.0
        input_list, scale = preprocess(image, self.image_size, camera, translation_scale_norm)

        boxes, scores, labels, rotations, translations = self.model.predict_on_batch(input_list)
        print(boxes, scores, labels, rotations, translations)


if __name__ == '__main__':
    efficientpose_detector = EfficientposeDetection(
        Path("/media/lars/Volume/Bachelor/Projekte/models/efficientpose/phi_3_occlusion_best_ADD(-S).h5"))
    path = "/media/lars/Volume/Bachelor/Projekte/datasets/Linemod_preprocessed/data/01/rgb/0000.png"
    img = Image.open(path)
    img = np.array(img)
    camera_k = np.array([[585.75607, 0, 320.5], \
                         [0, 585.75607, 240.5], \
                         [0, 0, 1, ]])
    print(efficientpose_detector.detect(img, camera_k))

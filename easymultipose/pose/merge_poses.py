from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import cosypose.utils.tensor_collection as tc
from cosypose.datasets.bop_object_datasets import BOPObjectDataset
from cosypose.integrated.multiview_predictor import MultiviewScenePredictor
from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.scripts.run_custom_scenario import read_csv_candidates, read_cameras
from easymultipose.pose.detection import Detection
from easymultipose.pose.pose_detection import PoseDetection


def prepare_candidates(candidates: Dict[str, List[Detection]]):
    view_ids = []
    scene_ids = []
    scores = []
    labels = []

    rotations = []
    translations = []

    for view_id in candidates:
        if candidates[view_id] is None:
            continue
        for detection in candidates[view_id]:
            view_ids.append(view_id)
            scene_ids.append(0)
            labels.append(detection.name)
            scores.append(detection.likelihood)
            pose = detection.pose
            rotations.append(pose[0:3, 0:3])
            translations.append(pose[0:3, 3])

    infos = pd.DataFrame(data={"view_id": view_ids, "scene_id": scene_ids, "score": scores, "label": labels})
    R = np.stack(rotations)
    t = np.stack(translations) * 1e-3
    R = torch.tensor(R, dtype=torch.float)
    t = torch.tensor(t, dtype=torch.float)
    TCO = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(len(R), 1, 1)
    TCO[:, :3, :3] = R
    TCO[:, :3, -1] = t
    candidates = tc.PandasTensorCollection(poses=TCO, infos=infos)
    return candidates


def prepare_cameras(cameras, view_ids):
    """cameras: {"<view_id>":[<camera parameters>] }"""
    all_K = []
    for view_id in view_ids:
        all_K.append(cameras[view_id])
    K = torch.as_tensor(np.stack(all_K))
    cameras = tc.PandasTensorCollection(K=K, infos=pd.DataFrame(dict(view_id=view_ids)))
    return cameras


def merge_poses(candidates, cameras, object_path):
    candidates = prepare_candidates(candidates).float().cuda()
    candidates.infos['group_id'] = 0

    view_ids = np.unique(candidates.infos['view_id'])
    cameras = prepare_cameras(cameras, view_ids).float().cuda()
    cameras.infos['scene_id'] = 0
    cameras.infos['batch_im_id'] = np.arange(len(view_ids))

    object_ds = BOPObjectDataset(object_path)
    mesh_db = MeshDataBase.from_object_ds(object_ds)

    mv_predictor = MultiviewScenePredictor(mesh_db)
    predictions = mv_predictor.predict_scene_state(candidates, cameras,
                                                   score_th=0.3,
                                                   use_known_camera_poses=False,
                                                   ransac_n_iter=2000,
                                                   ransac_dist_threshold=0.02,
                                                   ba_n_iter=10)
    objects = predictions['scene/objects']
    # cameras = predictions['scene/cameras']
    # reproj = predictions['ba_output']

    return objects


class MultiViewDetector:
    def __init__(self, detector: PoseDetection, object_path: Path):
        self.detector = detector
        self.object_path = object_path

    def detect(self, images: List, camera_ks: List):
        if len(images) != len(camera_ks):
            raise Exception("Same amount of images and camera parameters must be supplied!")
        return merge_poses([self.detector.detect(image, camera_k) for image, camera_k in zip(images, camera_ks)],
                           camera_ks, self.object_path)

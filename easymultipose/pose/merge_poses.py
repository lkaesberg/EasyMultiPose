import numpy as np
import pandas as pd
import torch
import cosypose.utils.tensor_collection as tc
from cosypose.datasets.bop_object_datasets import BOPObjectDataset
from cosypose.integrated.multiview_predictor import MultiviewScenePredictor
from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.scripts.run_custom_scenario import read_csv_candidates, read_cameras


def prepare_candidates(candidates):
    """candidates: {"<view_id>":{"<obj_name>": ([<mat4x4 pose>], <score>)  }"""
    view_ids = []
    scene_ids = []
    scores = []
    labels = []

    rotations = []
    translations = []

    for view_id in candidates:
        if candidates[view_id] is None:
            continue
        for obj_id in candidates[view_id]:
            view_ids.append(view_id)
            scene_ids.append(0)
            labels.append(obj_id)
            scores.append(candidates[view_id][obj_id][1])
            pose = np.array(candidates[view_id][obj_id][0])
            rotations.append(pose[0:3, 0:3])
            translations.append(pose[0:3, 3])

    infos = pd.DataFrame(data={"view_id": view_ids, "scene_id": scene_ids, "score": scores, "label": labels})
    print(infos)
    print()
    R = np.stack(rotations)
    t = np.stack(translations) * 1e-3
    print(R)
    print()
    print(t)
    print()
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
    cameras = predictions['scene/cameras']
    reproj = predictions['ba_output']

    print("------------------- OBJ")
    print(objects)
    print("------------------- CAM")
    print(cameras)
    print("------------------- REP")
    print(reproj)

    return None

import numpy as np
import pandas as pd
import torch
import cosypose.utils.tensor_collection as tc


def prepare_candidates(candidates):
    """candidates: {"<view_id>":{"<obj_name>": ([<mat4x4 pose>], <score>)  }"""
    view_ids = []
    scene_ids = []
    scores = []
    labels = []

    rotations = []
    translations = []

    for view_id in candidates:
        for obj_id in candidates[view_id]:
            view_ids.append(view_id)
            scene_ids.append(0)
            labels.append(obj_id)
            scores.append(candidates[view_id][obj_id][1])

            rotations.append(candidates[view_id][obj_id][0][0:3, 0:3])
            translations.append(candidates[view_id][obj_id][0][0:3, 3])

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


def merge_poses(candidates, cameras):
    candidates = prepare_candidates(candidates)
    view_ids = np.unique(candidates.infos['view_id'])
    cameras = prepare_cameras(cameras, view_ids)

    return None

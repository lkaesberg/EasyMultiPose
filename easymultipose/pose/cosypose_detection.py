from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from cosypose.datasets.bop_object_datasets import BOPObjectDataset
from cosypose.integrated.detector import Detector
from cosypose.integrated.pose_predictor import CoarseRefinePosePredictor
from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from cosypose.scripts.convert_models_to_urdf import convert_obj_dataset_to_urdfs_abs_path
from cosypose.training.detector_models_cfg import check_update_config as check_update_config_detector
from cosypose.training.detector_models_cfg import create_model_detector
from cosypose.training.pose_models_cfg import check_update_config as check_update_config_pose, \
    create_model_refiner, \
    create_model_coarse

from easymultipose.pose.pose_detection import PoseDetection
from easymultipose.urdf_cfg import set_urdf_path

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_detector(path):
    cfg = yaml.load((path / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config_detector(cfg)
    label_to_category_id = cfg.label_to_category_id
    model = create_model_detector(cfg, len(label_to_category_id))
    ckpt = torch.load(path / 'checkpoint.pth.tar')
    ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model = model.cuda().eval()
    model.cfg = cfg
    model.config = cfg
    model = Detector(model)
    return model


def load_pose_models(models_path, coarse_path, refiner_path=None, n_workers=8):
    cfg_path = coarse_path / 'config.yaml'
    cfg = yaml.load(cfg_path.read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config_pose(cfg)
    object_ds = BOPObjectDataset(models_path)
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    renderer = BulletBatchRenderer(object_set=cfg.urdf_ds_name, n_workers=n_workers)
    mesh_db_batched = mesh_db.batched().cuda()

    def load_model(path):
        if path is None:
            return
        cfg_path = path / 'config.yaml'
        cfg = yaml.load(cfg_path.read_text(), Loader=yaml.FullLoader)
        cfg = check_update_config_pose(cfg)
        if cfg.train_refiner:
            model = create_model_refiner(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        else:
            model = create_model_coarse(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        ckpt_path = path / 'checkpoint.pth.tar'
        ckpt = torch.load(ckpt_path)
        ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt)
        model = model.cuda().eval()
        model.cfg = cfg
        model.config = cfg
        return model

    coarse_model = load_model(coarse_path)
    refiner_model = load_model(refiner_path)
    model = CoarseRefinePosePredictor(coarse_model=coarse_model,
                                      refiner_model=refiner_model)
    return model, mesh_db


def inference(detector, pose_predictor, image, camera_k):
    # [1,540,720,3]->[1,3,540,720]
    images = torch.from_numpy(image).cuda().float().unsqueeze_(0)
    images = images.permute(0, 3, 1, 2) / 255
    # [1,3,3]
    cameras_k = torch.from_numpy(camera_k).cuda().float().unsqueeze_(0)
    # 2D detector
    # print("start detect object.")
    box_detections = detector.get_detections(images=images, one_instance_per_class=False,
                                             detection_th=0.8, output_masks=False, mask_th=0.9)
    # pose esitimition
    if len(box_detections) == 0:
        return None
    # print("start estimate pose.")
    final_preds, all_preds = pose_predictor.get_predictions(images, cameras_k, detections=box_detections,
                                                            n_coarse_iterations=1, n_refiner_iterations=4)
    # print("inference successfully.")
    # result: this_batch_detections, final_preds
    return final_preds.cpu()


class CosyposeDetection(PoseDetection):

    def __init__(self, models_path, detector_path, coarse_path, refiner_path=None):
        models_path = Path(models_path)
        urdf_path = models_path / 'urdfs'
        if not urdf_path.exists():
            convert_obj_dataset_to_urdfs_abs_path(models_path)
        set_urdf_path(urdf_path)
        self.detector = load_detector(Path(detector_path))
        if refiner_path:
            self.model, _ = load_pose_models(models_path, Path(coarse_path), Path(refiner_path), 4)
        else:
            self.model, _ = load_pose_models(models_path, Path(coarse_path), Path(refiner_path), 4)

    def detect(self, image, camera):
        data = inference(self.detector, self.model, image, camera)
        predictions = {}
        if not data:
            return None
        for i in range(len(data)):
            pose = data.poses[i].numpy()
            predictions[data.infos.iloc[i].label] = (pose, data.infos.iloc[i].score)
        return predictions


if __name__ == '__main__':
    cosypose_detector = CosyposeDetection(
        models_path="/media/lars/Volume/Bachelor/Projekte/cosypose/local_data/bop_datasets/ycbv/models",
        detector_path="/media/lars/Volume/Bachelor/Projekte/models/test_det",
        coarse_path="/media/lars/Volume/Bachelor/Projekte/cosypose/local_data/experiments/coarse-bop-ycbv-pbr--724183",
        refiner_path="/media/lars/Volume/Bachelor/Projekte/cosypose/local_data/experiments/refiner-bop-ycbv-pbr--604090")
    path = "/home/lars/Downloads/csm_fertighaus-bauen-startseiten-bild_d13e0ec91d.jpg"
    img = Image.open(path)
    img = np.array(img)
    camera_k = np.array([[585.75607, 0, 320.5], \
                         [0, 585.75607, 240.5], \
                         [0, 0, 1, ]])
    print(cosypose_detector.detect(img, camera_k))

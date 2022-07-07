import argparse
from pathlib import Path

from cosypose.scripts.convert_models_to_urdf import convert_obj_dataset_to_urdfs_abs_path
from cosypose.training.train_pose import train_pose

from easymultipose.urdf_cfg import set_urdf_path


def train_cosypose():
    cfg = argparse.ArgumentParser('').parse_args([])

    cfg.n_symmetries_batch = 32

    cfg.val_epoch_interval = 10
    cfg.test_ds_names = []
    cfg.test_epoch_interval = 30
    cfg.n_test_frames = None

    cfg.input_resize = (480, 640)
    cfg.rgb_augmentation = True
    cfg.background_augmentation = True
    cfg.gray_augmentation = False

    # Model
    cfg.backbone_str = 'efficientnet-b3'
    cfg.run_id_pretrain = None
    cfg.n_pose_dims = 9
    cfg.n_rendering_workers = 0
    cfg.refiner_run_id_for_test = None
    cfg.coarse_run_id_for_test = None

    # Optimizer
    cfg.lr = 3e-4
    cfg.weight_decay = 0.
    cfg.n_epochs_warmup = 50
    cfg.lr_epoch_decay = 500
    cfg.clip_grad_norm = 0.5

    # Training
    cfg.batch_size = 10
    cfg.epoch_size = 10000
    cfg.n_epochs = 700
    cfg.n_dataloader_workers = 0

    # Method
    cfg.loss_disentangled = True
    cfg.n_points_loss = 1000
    cfg.TCO_input_generator = 'fixed'
    cfg.n_iterations = 1
    cfg.min_area = None

    cfg.resume_run_id = None

    cfg.object_ds_name = Path(
        "/home/lars/Unity/6dposeestimation-datasetprovider/bop_dataset/models")

    urdf_path = cfg.object_ds_name / 'urdfs'
    if not urdf_path.exists():
        convert_obj_dataset_to_urdfs_abs_path(cfg.object_ds_name)

    cfg.urdf_ds_name = urdf_path
    set_urdf_path(cfg.urdf_ds_name)

    cfg.train_ds_names = [
        (Path("/home/lars/Unity/6dposeestimation-datasetprovider/bop_dataset"), "train", 1)]
    cfg.val_ds_names = cfg.train_ds_names

    cfg.run_id = Path("/home/lars/Python/model_cosy")

    cfg.voc_folder = Path("/media/lars/Volume/Bachelor/VOCdevkit/VOC2012")

    train_pose(cfg)


if __name__ == '__main__':
    train_cosypose()

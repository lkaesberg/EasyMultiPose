import argparse
from pathlib import Path

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
    cfg.n_points_loss = 2600
    cfg.TCO_input_generator = 'fixed'
    cfg.n_iterations = 1
    cfg.min_area = None

    cfg.resume_run_id = None

    cfg.object_ds_name = Path(
        "/media/lars/Volume/Bachelor/Projekte/cosypose/local_data/bop_datasets/tless/models_cad")

    cfg.urdf_ds_name = Path("/media/lars/Volume/Bachelor/Projekte/cosypose/local_data/urdfs/tless.cad")
    set_urdf_path(cfg.urdf_ds_name)

    cfg.train_ds_names = [
        (Path("/media/lars/Volume/Bachelor/Projekte/cosypose/local_data/bop_datasets/tless"), "train_primesense", 1)]
    cfg.val_ds_names = cfg.train_ds_names

    cfg.run_id = Path("/media/lars/Volume/Bachelor/Projekte/models/test")

    cfg.voc_folder = Path("/media/lars/Volume/Bachelor/Projekte/cosypose/local_data/VOCdevkit/VOC2012")

    train_pose(cfg)


if __name__ == '__main__':
    train_cosypose()

import argparse
import numpy as np
import os
from colorama import Fore, Style
from pathlib import Path

from cosypose.training.train_detector import train_detector
from cosypose.utils.logging import get_logger

logger = get_logger(__name__)


def train_cosypose_detector():
    cfg = argparse.ArgumentParser('').parse_args([])

    N_CPUS = int(os.environ.get('N_CPUS', 10))
    N_GPUS = int(os.environ.get('N_PROCS', 1))
    N_WORKERS = 0
    N_RAND = np.random.randint(1e6)
    cfg.n_gpus = N_GPUS

    run_comment = ''

    # Data
    cfg.val_epoch_interval = 10
    cfg.test_ds_names = []
    cfg.test_epoch_interval = 30
    cfg.n_test_frames = None

    cfg.input_resize = (480, 640)
    cfg.rgb_augmentation = True
    cfg.background_augmentation = True
    cfg.gray_augmentation = False

    # Model
    cfg.backbone_str = 'resnet50-fpn'
    cfg.anchor_sizes = ((32,), (64,), (128,), (256,), (512,))

    # Pretraning
    cfg.run_id_pretrain = None
    cfg.pretrain_coco = True

    # Training
    cfg.batch_size = 2
    cfg.epoch_size = 5000
    cfg.n_epochs = 600
    cfg.lr_epoch_decay = 200
    cfg.n_epochs_warmup = 50
    cfg.n_dataloader_workers = N_WORKERS

    # Optimizer
    cfg.optimizer = 'sgd'
    cfg.lr = (0.02 / 8) * N_GPUS * float(cfg.batch_size / 4)
    cfg.weight_decay = 1e-4
    cfg.momentum = 0.9

    # Method
    cfg.rpn_box_reg_alpha = 1
    cfg.objectness_alpha = 1
    cfg.classifier_alpha = 1
    cfg.mask_alpha = 1
    cfg.box_reg_alpha = 1

    N_GPUS = int(os.environ.get('N_PROCS', 1))
    cfg.epoch_size = cfg.epoch_size // N_GPUS

    cfg.resume_run_id = None

    cfg.train_ds_names = [
        (Path("/home/lars/Unity/6dposeestimation-datasetprovider/bop_dataset"), "train", 3)]
    cfg.val_ds_names = cfg.train_ds_names

    cfg.run_id = Path("/home/lars/Python/model")

    cfg.voc_folder = Path("/media/lars/Volume/Bachelor/VOCdevkit/VOC2012")

    train_detector(cfg)


if __name__ == '__main__':
    train_cosypose_detector()

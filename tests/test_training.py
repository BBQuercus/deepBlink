"""Unittests for the deepblink.training module."""

import glob
import os

import numpy as np

from deepblink.io import load_model
from deepblink.training import run_experiment


def _config(savedir: str, dataset: str) -> dict:
    """Smallest configuration that trains a U-Net for one epoch."""
    return {
        "name": "test",
        "run_name": "test",
        "use_wandb": False,
        "savedir": savedir,
        "augmentation_args": dict.fromkeys(["flip", "illuminate", "rotate", "gaussian_noise", "translate"], False),
        "dataset": "SpotsDataset",
        "dataset_args": {"name": dataset, "cell_size": 4, "smooth_factor": 1},
        "model": "SpotsModel",
        "network": "unet",
        "network_args": {"dropout": 0.3, "filters": 2, "ndown": 1, "l2": 1e-6, "block": "convolutional"},
        "loss": "combined_dice_rmse",
        "optimizer": "amsgrad",
        "train_args": {"batch_size": 2, "epochs": 1, "learning_rate": 1e-4, "overfit": False},
    }


def test_train_save_load_finetune(tmp_path):
    """A saved checkpoint must load again and continue training (the pre_train path)."""
    rng = np.random.default_rng(0)
    images = rng.random((4, 32, 32)).astype(np.float32)
    labels = np.array([rng.uniform(4, 28, size=(3, 2)) for _ in images], dtype=object)
    dataset = str(tmp_path / "data.npz")
    np.savez(
        dataset, x_train=images, y_train=labels, x_valid=images, y_valid=labels, x_test=images, y_test=labels
    )

    run_experiment(_config(str(tmp_path), dataset))
    (checkpoint,) = glob.glob(os.path.join(tmp_path, "*.h5"))

    run_experiment(_config(str(tmp_path), dataset), pre_model=load_model(checkpoint))

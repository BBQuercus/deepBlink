"""CLI submodule for configuration."""

import logging
import os
import yaml

from ..io import securename
from ._train import _get_values


class HandleConfig:
    """Handle configuration submodule for CLI.

    Args:
        arg_name: Name of output file.
        arg_raw: If "description" / "value" should be removed.
        logger: Logger to log verbose output.
    """

    def __init__(self, arg_name: str, arg_raw: bool, logger: logging.Logger):
        self.raw = arg_raw
        self.logger = logger
        self.logger.info(f"\U00002699 starting config submodule, raw {arg_raw}")

        self.abs_output = os.path.abspath(securename(arg_name) + ".yaml")

    def __call__(self):
        """Save configuration as yaml file."""
        self.save_yaml()
        self.logger.info(f"\U0001F3C1 saved config file to {self.abs_output}")

    @property
    def config(self):
        """Default configuration as dictionary."""
        configuration = {
            "name": {"description": "Wandb/model name", "value": "deepBlink"},
            "run_name": {"description": "Current run", "value": "deepBlink_is_sweet"},
            "savedir": {"description": "Model saving path", "value": "PATH/TO/OUTDIR"},
            "use_wandb": {"description": "If wandb.ai should be used", "value": False},
            "augmentation_args": {
                "flip": {"description": "Flipping", "value": False},
                "illuminate": {"description": "Illumination changes", "value": False},
                "rotate": {"description": "Rotation", "value": False},
                "gaussian_noise": {"description": "Gaussian_noise", "value": False},
                "translate": {"description": "Translation", "value": False},
            },
            "dataset": {"description": "Dataset class", "value": "SpotsDataset"},
            "dataset_args": {
                "name": {
                    "description": "Dataset.npz file",
                    "value": "PATH/TO/DATASET.NPZ",
                },
                "cell_size": {
                    "description": "Size of one cell in the grid",
                    "value": 4,
                },
                "smooth_factor": {
                    "description": "True label smoothing factor",
                    "value": 1,
                },
            },
            "model": {"description": "Name of the model class", "value": "SpotsModel"},
            "network": {
                "description": "Network architecture function",
                "value": "unet",
            },
            "network_args": {
                "dropout": {
                    "description": "Fraction of input units to drop",
                    "value": 0.3,
                },
                "filters": {
                    "description": "2 ** filters in the first layer",
                    "value": 5,
                },
                "ndown": {"description": "Network depth (encoding steps)", "value": 2},
                "l2": {"description": "L2 regularization factor", "value": 1e-6},
                "block": {
                    "description": "Basic architectural block",
                    "value": "convolutional",
                },
            },
            "loss": {"description": "Loss function", "value": "combined_dice_rmse"},
            "optimizer": {"description": "Optimizer function", "value": "amsgrad"},
            "train_args": {
                "batch_size": {"description": "Samples per mini-batch", "value": 2},
                "epochs": {"description": "Total rounds of training", "value": 200},
                "learning_rate": {"description": "Learning rate", "value": 1e-4},
                "overfit": {"description": "Single batch overfitting", "value": False},
            },
        }

        if self.raw:
            configuration = _get_values(configuration)

        return configuration

    def save_yaml(self):
        """Dump configuration into yaml file."""
        with open(self.abs_output, "w") as outfile:
            yaml.dump(self.config, outfile, default_flow_style=False)

"""CLI submodule for configuration."""

import logging
import os
import yaml

from ..io import securename


class HandleConfig:
    """Handle configuration submodule for CLI.

    Args:
        arg_output: Name of output file.
        logger: Logger to log verbose output.
    """

    def __init__(self, arg_name: str, logger: logging.Logger):
        self.logger = logger
        self.logger.info("\U00002699 starting config submodule")

        self.abs_output = os.path.abspath(securename(arg_name) + ".yaml")

    def __call__(self):
        """Save configuration as yaml file."""
        self.save_yaml()
        self.logger.info(f"\U0001F3C1 saved config file to {self.abs_output}")

    @property
    def config(self):
        """Default configuration as dictionary."""
        return {
            "name": {
                "description": "Name of the Wandb/model project",
                "value": "deepBlink",
            },
            "run_name": {
                "description": "Name of a specific run",
                "value": "deepBlink_run",
            },
            "savedir": {
                "description": "Path to where the model should be saved",
                "value": "PATH/TO/OUTDIR",
            },
            "use_wandb": {
                "description": "Boolean variable to specify if Wandb should be used",
                "value": False,
            },
            "dataset": {
                "description": "Name of dataset class",
                "value": "SpotsDataset",
            },
            "dataset_args": {
                "version": {
                    "description": "Path to dataset.npz file",
                    "value": "PATH/TO/DATASET.NPZ",
                },
                "cell_size": {
                    "description": "Size of one cell in the grid",
                    "value": 4,
                },
                "flip": {
                    "description": "If flipping should be used as augmentation",
                    "value": False,
                },
                "illuminate": {
                    "description": "If illuminate should be used as augmentation",
                    "value": False,
                },
                "rotate": {
                    "description": "If rotate should be used as augmentation",
                    "value": False,
                },
                "gaussian_noise": {
                    "description": "If gaussian_noise should be used as augmentation",
                    "value": False,
                },
                "translate": {
                    "description": "If translate should be used as augmentation",
                    "value": False,
                },
            },
            "model": {"description": "Name of the model class", "value": "SpotsModel"},
            "network": {
                "description": "Name of the network architecture",
                "value": "convolution",
            },
            "network_args": {
                "dropout": {
                    "description": "Percentage of dropout only for resnet architecture",
                    "value": 0.3,
                },
                "filters": {
                    "description": "log2 number of filters in the first convolution layers",
                    "value": 5,
                },
                "n_extra_down": {
                    "description": "Number of further down steps in the UNet",
                    "value": 0,
                },
            },
            "loss": {"description": "Primary loss", "value": "combined_dice_rmse"},
            "optimizer": {"description": "Optimizer", "value": "amsgrad"},
            "train_args": {
                "batch_size": {
                    "description": "Number of images per mini-batch",
                    "value": 2,
                },
                "epochs": {"description": "Total rounds of training", "value": 200},
                "learning_rate": {"description": "Learning rate", "value": 1e-4},
                "overfit": {
                    "description": "If model should overfit to one batch",
                    "value": False,
                },
            },
        }

    def save_yaml(self):
        """Dump configuration into yaml file."""
        with open(self.abs_output, "w") as outfile:
            yaml.dump(self.config, outfile, default_flow_style=False)

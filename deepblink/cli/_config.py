"""CLI submodule for configuration."""

import argparse
import logging
import os
import yaml

from ..io import securename
from ._parseutil import CustomFormatter
from ._parseutil import _add_utils


def _parse_args_config(
    subparsers: argparse._SubParsersAction, parent_parser: argparse.ArgumentParser
):
    """Subparser for configuration."""
    parser = subparsers.add_parser(
        "config",
        parents=[parent_parser],
        add_help=False,
        formatter_class=CustomFormatter,
        description=(
            "\U0001F528 Configuration submodule \U0001F528\n\n"
            "Prepare a configuration file used to adjust parameters during training. "
        ),
        help="\U0001F528 Create a configuration file for training.",
    )
    group2 = parser.add_argument_group("Optional")
    group2.add_argument(
        "-n",
        "--name",
        type=str,
        default="config",
        help=(
            "Custom configuration name. "
            'The file extension "yaml" will be added automatically to the given name.'
            '[default: "config"]'
        ),
    )
    _add_utils(parser)


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

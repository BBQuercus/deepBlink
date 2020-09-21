"""CLI submodule for configuration."""

import argparse
import logging
import os
import yaml

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
        description="\U00002699 Configuration submodule",
        help="\U00002699 create a configuration file for training",
    )
    group2 = parser.add_argument_group("Optional")
    group2.add_argument(
        "-n",
        "--name",
        type=str,
        default="config",
        help="custom name of configuration file. file extension will be added automatically [default: config]",
    )
    _add_utils(parser)


class HandleConfig:
    """Handle configuration submodule for CLI.

    Args:
        arg_output: Name of output file.
        logger: Logger to log verbose output.
    """

    def __init__(self, arg_name: str, logger: logging.Logger):
        self.name = arg_name
        self.logger = logger
        self.logger.info("\U00002699 starting config submodule")

        self.abs_output = os.path.abspath(self.name + ".yaml")

    def __call__(self):
        """Save configuration as yaml file."""
        self.save_yaml()
        self.logger.info(f"\U0001F3C1 saved config file to {self.abs_output}")

    @property
    def config(self):
        """Default configuration as dictionary."""
        return {
            "name": "deepBlink",
            "savedir": "PATH/TO/OUTDIR",
            "comments": "COMMENT ON WANDB",
            "use_wandb": False,
            "dataset": "SpotsDataset",
            "dataset_args": {
                "version": "PATH/TO/DATASET.NPZ",
                "cell_size": 4,
                "flip": False,
                "illuminate": False,
                "rotate": False,
                "gaussian_noise": False,
                "translate": False,
            },
            "model": "SpotsModel",
            "network": "inception_squeeze",
            "network_args": {"dropout": 0.0, "filters": 4, "n_extra_down": 0},
            "loss": "combined_bce_rmse",
            "optimizer": "adam",
            "train_args": {
                "batch_size": 2,
                "epochs": 1000,
                "learning_rate": 1e-4,
                "overfit": False,
            },
        }

    def save_yaml(self):
        """Dump configuration into yaml file."""
        with open(self.abs_output, "w") as outfile:
            yaml.dump(self.config, outfile, default_flow_style=False)

"""CLI submodule for training."""

import argparse
import logging
import os
import yaml

from ..training import run_experiment
from ._parseutil import CustomFormatter
from ._parseutil import FMT
from ._parseutil import FileType
from ._parseutil import _add_utils


def _parse_args_train(
    subparsers: argparse._SubParsersAction, parent_parser: argparse.ArgumentParser,
):
    """Subparser for training."""
    parser = subparsers.add_parser(
        "train",
        parents=[parent_parser],
        formatter_class=CustomFormatter,
        add_help=False,
        description=(
            f"\U0001F686 {FMT.dc}Training submodule{FMT.e} \U0001F686\n\n"
            f"Train a custom model using a custom dataset created in {FMT.b}deepblink create{FMT.e} "
            "or using a published dataset."
        ),
        help="\U0001F686 Train a freshly baked model on a dataset.",
    )
    group1 = parser.add_argument_group(f"{FMT.r}Required{FMT.e}")
    group1.add_argument(
        "-c",
        "--config",
        type=FileType("yaml"),
        required=True,
        help=(
            "Configuration file. "
            f"Path to the config.yaml created using {FMT.b}deepblink config{FMT.e}. "
            f"The path be relative (e.g. {FMT.b}../dir{FMT.e}) or absolute (e.g. {FMT.b}/Users/myname/{FMT.e}. "
            "Please see the training information on the wiki to configure the file to your requirements. "
            "[required]"
        ),
    )
    group2 = parser.add_argument_group(f"{FMT.g}Optional{FMT.e}")
    group2.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=None,
        help=(
            "GPU index. "
            "Value passed CUDA_VISIBLE_DEVICES if a GPU is used for training. "
            "[default: None]"
        ),
    )
    _add_utils(parser)


class HandleTrain:
    """Handle checking submodule for CLI.

    Args:
        arg_config: Path to config.yaml file.
        arg_gpu: Which gpu is to be used.
        logger: Verbose logger.
    """

    def __init__(self, arg_config: str, arg_gpu: int, logger: logging.Logger):
        self.raw_config = arg_config
        self.gpu = arg_gpu
        self.logger = logger
        self.logger.info("\U0001F686 starting checking submodule")

    def __call__(self):
        """Set configuration and start training loop."""
        self.set_gpu()
        self.logger.info("\U0001F3C3 beginning with training")
        run_experiment(self.config)
        self.logger.info("\U0001F3C1 training complete")

    @property
    def config(self):
        """Load config.yaml file into memory."""
        if not os.path.isfile(self.raw_config):
            raise ImportError(
                "\U0000274C Input file does not exist. Please provide a valid path."
            )
        if not self.raw_config.lower().endswith("yaml"):
            raise ImportError(
                "\U0000274C Input file extension invalid. Please provide the filetype yaml."
            )
        with open(self.raw_config, "r") as config_file:
            config = yaml.safe_load(config_file)
        self.logger.info(f"\U0001F4C2 loaded config file: {config}")
        return config

    def set_gpu(self):
        """Set GPU environment variable."""
        if self.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{self.gpu}"
        self.logger.info(f"\U0001F5A5 set GPU number to {self.gpu}")

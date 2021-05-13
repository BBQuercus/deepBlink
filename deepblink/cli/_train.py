"""CLI submodule for training."""

import logging
import os
import yaml

from ..io import load_model
from ..training import run_experiment


def _get_values(dct: dict) -> dict:
    """Remove description / value metadata from dictionary recursively."""
    return {
        k: v["value"]
        if isinstance(v, dict) and "value" in v
        else _get_values(v)
        if isinstance(v, dict)
        else v
        for k, v in dct.items()
    }


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
        self.logger.info(f"\U0001F4C2 loaded config file: {self. config}")
        self.logger.info("\U0001F3C3 beginning with training")
        run_experiment(self.config, pre_model=self.model)
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

        config = _get_values(config)
        return config

    @property
    def model(self):
        """Load pre-trained model if defined."""
        pre_train = self.config["train_args"]["pre_train"]
        if pre_train is not None:
            try:
                model = load_model(pre_train)
                self.logger.info(
                    f"Pre-trained model found and loaded - using {pre_train}."
                )
                return model
            except (ValueError, ImportError):
                self.logger.info(
                    f"Pre-trained model not found or faulty - {pre_train} could not be imported. "
                    "Training from scratch."
                )
        return None

    def set_gpu(self):
        """Set GPU environment variable."""
        if self.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{self.gpu}"
        self.logger.info(f"\U0001F5A5 set GPU number to {self.gpu}")

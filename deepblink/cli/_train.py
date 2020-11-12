"""CLI submodule for training."""

import logging
import os
import yaml

from ..training import run_experiment


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

        def _get_values(dct):
            return {
                k: v["value"] if "value" in v else _get_values(v)
                for k, v in dct.items()
            }

        config = _get_values(config)

        self.logger.info(f"\U0001F4C2 loaded config file: {config}")
        return config

    def set_gpu(self):
        """Set GPU environment variable."""
        if self.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{self.gpu}"
        self.logger.info(f"\U0001F5A5 set GPU number to {self.gpu}")

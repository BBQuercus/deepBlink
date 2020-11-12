"""CLI submodule for checking image shapes."""

import logging
import os
import textwrap

from ..io import load_image
from ..util import predict_shape


class HandleCheck:
    """Handle checking submodule for CLI.

    Args:
        arg_input: Path to image.
        logger: Verbose logger.
    """

    def __init__(self, arg_input: str, logger: logging.Logger):
        self.raw_input = arg_input
        self.logger = logger
        self.logger.info("\U0001F537 starting checking submodule")

        self.abs_input = os.path.abspath(self.raw_input)

    def __call__(self) -> None:
        """Run check for input image."""
        print(
            textwrap.dedent(
                f"""
        1. Your image has a shape of: {self.image.shape}
        ----------
        2. Possible parameters
        \U000027A1 x, y: single 2D image used for one prediction
        \U000027A1 z: third (height) dimension
        \U000027A1 c: color channels
        \U000027A1 t: time dimension
        \U000027A1 3: RGB color stack
        ----------
        3. By default we would assign: "({predict_shape(self.image.shape)})"
        \U0001F449 If this is incorrect, please provide the proper shape using the --shape flag to the
        submodule predict in deepblink's command line interface
        """
            )
        )

    @property
    def image(self):
        """Load a single image."""
        return load_image(self.abs_input)

"""CLI submodule for checking image shapes."""

import argparse
import logging
import os
import textwrap

from ..io import EXTENSIONS
from ..io import load_image
from ..util import predict_shape
from ._parseutil import CustomFormatter
from ._parseutil import FileType
from ._parseutil import _add_utils


def _parse_args_check(
    subparsers: argparse._SubParsersAction, parent_parser: argparse.ArgumentParser,
):
    """Subparser for checking."""
    parser = subparsers.add_parser(
        "check",
        parents=[parent_parser],
        formatter_class=CustomFormatter,
        add_help=False,
        description="\U0001F537 Checking submodule",
        help="\U0001F537 determine your input images' shape",
    )
    group1 = parser.add_argument_group("Required")
    group1.add_argument(
        "INPUT",
        type=FileType(EXTENSIONS),
        help=f"input image location [required] [filetypes: {EXTENSIONS}]",
    )
    _add_utils(parser)


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

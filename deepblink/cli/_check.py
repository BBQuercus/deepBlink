"""CLI submodule for checking image shapes."""

import argparse
import logging
import os
import textwrap

from ..io import EXTENSIONS
from ..io import load_image
from ..util import predict_shape
from ._parseutil import CustomFormatter
from ._parseutil import FMT
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
        description=(
            f"\U0001F537 {FMT.dc}Checking submodule{FMT.e} \U0001F537\n\n"
            "Check the arrangement of your image's axis also known as image shape. "
        ),
        help="\U0001F537 Determine your input image's shape.",
    )
    group1 = parser.add_argument_group(f"{FMT.r}Required{FMT.e}")
    group1.add_argument(
        "INPUT",
        type=FileType(EXTENSIONS),
        help=(
            "Input image. "
            "Path to the image file to be checked. "
            f"The path be relative (e.g. {FMT.b}../dir{FMT.e}) or absolute (e.g. {FMT.b}/Users/myname/{FMT.e}. "
            "Input can either be given as path to a directory containing files or as a single file. "
            "Note that only the specified filetypes will be processed. "
            f"[required] [filetypes: {', '.join(EXTENSIONS)}]"
        ),
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

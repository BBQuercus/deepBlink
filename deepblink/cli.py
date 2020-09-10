"""Module that contains the command line app.

Used for inferencing of new data with pretrained models.
Because the model is loaded into memory for every run, it is faster to
run multiple images at once by passing a directory as input.

Usage:
    ``$ deepblink [-h] [-o OUTPUT] [-t {csv,txt}] [-r RADIUS] [-v] [-V] MODEL INPUT``
    ``$ deepblink --help``

Positional Arguments:
    MODEL           model .h5 file location
    INPUT           input file/folder location

Optional Arguments:
    -h, --help      show this help screen

    -o, --output    output file/folder location [default: input location]
    -t, --type      output file type [options: csv, txt] [default: csv]
    -r, --radius    if given, calculate the integrated intensity in the given radius around each coordinate
    -v, --verbose   set program output to verbose [default: quiet]
    -V, --version   show version program's version number and exit
"""
import argparse
import logging
import os
import sys
import textwrap

import numpy as np

from .inference import predict
from .inference import get_intensities
from .io import basename
from .io import grab_files
from .io import load_image
from .io import load_model


# Configure verbose logger
logging.basicConfig(
    format="%(asctime)s: %(message)s", stream=sys.stdout, level=logging.INFO
)

# Removes tensorflow's information on CPU / GPU availablity.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# List of currently supported image file extensions.
EXTENSIONS = ["tif", "jpeg", "jpg", "png"]


class FileFolderType:
    """Custom type supporting folders or files."""

    def __init__(self):
        pass

    def __call__(self, value):  # noqa: D102
        """Python type internal function called by argparse to check input."""
        if not any((os.path.isdir(value), os.path.isfile(value))):
            raise argparse.ArgumentTypeError(
                f"Input value must be file or folder. '{value}' is not."
            )
        return value


class FolderType:
    """Custom type supporting folders."""

    def __init__(self):
        pass

    def __call__(self, value):  # noqa: D102
        """Python type internal function called by argparse to check input."""
        if not os.path.isdir(value):
            raise argparse.ArgumentTypeError(
                f"Input value must be folder and must exist. '{value}' is not."
            )
        return value


def _parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(
        prog="deepblink",
        description="deepBlinks command line interface \U0001F469\U0000200D\U0001F4BB for training, inferencing, and evaluation",
        epilog="We hope you enjoy using deepBlink \U0001F603",
    )
    parser.add_argument("-V", "--version", action="version", version="%(prog)s 0.0.6")
    subparsers = parser.add_subparsers(dest="command")

    # Train parser
    parser_train = subparsers.add_parser(
        "train", help="\U0001F35E train a freshly baked model on a dataset",
    )
    parser_train.add_argument(
        "INPUT",
        type=FileFolderType(),
        help=f"input file/folder location [filetypes: {EXTENSIONS}]",
    )

    # Check parser
    parser_check = subparsers.add_parser(
        "check", help="\U0001F537 \U0001F535 determine your input images' shape"
    )
    parser_check.add_argument(
        "INPUT",
        type=FileFolderType(),
        help=f"input file/folder location [filetypes: {EXTENSIONS}]",
    )

    # Predict parser
    parser_predict = subparsers.add_parser(
        "predict",
        help="\U0001F914 inference / prediction of data with a pre-trained model",
    )
    parser_predict.add_argument(
        "MODEL", type=argparse.FileType("r"), help="model .h5 file location"
    )
    parser_predict.add_argument(
        "INPUT",
        type=FileFolderType(),
        help=f"input file/folder location [filetypes: {EXTENSIONS}]",
    )
    parser_predict.add_argument(
        "-o",
        "--output",
        type=FolderType(),
        help="output file/folder location [default: input location]",
    )
    parser_predict.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["csv", "txt"],
        default="csv",
        help="output file type [options: csv, txt] [default: csv]",
    )
    parser_predict.add_argument(
        "-r",
        "--radius",
        type=int,
        default=None,
        help=textwrap.dedent(
            """if given, calculate the integrated intensity
        in the given radius around each coordinate. set radius to zero if only the
        central pixels intensity should be calculated."""
        ),
    )
    parser_predict.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="set program output to verbose [default: quiet]",
    )

    # Eval parser
    parser_eval = subparsers.add_parser(
        "eval", help="\U0001F3AD measure a models performance on a dataset"
    )
    parser_eval.add_argument(
        "INPUT",
        type=FileFolderType(),
        help=f"input file/folder location [filetypes: {EXTENSIONS}]",
    )

    args = parser.parse_args()
    return args


def main():
    """Entrypoint for the CLI."""
    args = _parse_args()

    logger = logging.getLogger("Verbose output logger")

    handler = HandlePredict(
        arg_model=args.MODEL.name,
        arg_input=args.INPUT,
        arg_output=args.output,
        arg_radius=args.radius,
        arg_type=args.type,
        arg_verbose=args.verbose,
        logger=logger,
    )

    handler.run()


class HandlePredict:
    """Handling of prediction submodule for CLI (check argparser for docstring).

    Args:
        arg_model: Path to model.h5 file.
        arg_input: Path to image file / folder with images.
        arg_output: Path to output directory.
        arg_radius: Size of integrated image intensity calculation.
        arg_type: Output file type.
        arg_verbose: If output is verbose.
        logger: Logger to log verbose output.
    """

    def __init__(
        self,
        arg_model: str,
        arg_input: str,
        arg_output: str,
        arg_radius: int,
        arg_type: str,
        arg_verbose: bool,
        logger: logging.Logger,
    ):
        self.fname_model = arg_model
        self.raw_input = arg_input
        self.raw_output = arg_output
        self.radius = arg_radius
        self.type = arg_type
        self.verbose = arg_verbose
        self.logger = logger

        self.extensions = EXTENSIONS
        self.abs_input = os.path.abspath(arg_input)

    @property
    def path_input(self):
        """Return absolute input path (dependent on file/folder input)."""
        if os.path.isdir(self.abs_input):
            path_input = self.abs_input
        elif os.path.isfile(self.abs_input):
            path_input = os.path.dirname(self.abs_input)
        return path_input

    @property
    def file_list(self):
        """Return a list with all files to be processed."""
        if os.path.isdir(self.abs_input):
            file_list = grab_files(self.abs_input, self.extensions)
        elif os.path.isfile(self.abs_input):
            file_list = [self.abs_input]
        else:
            raise ImportError(
                "\U0000274C Input file(s) could not be found. Please make sure all files exist."
            )
        return file_list

    @property
    def path_output(self):
        """Return the absolute output path (dependent if given)."""
        if os.path.exists(str(self.raw_output)):
            outpath = os.path.abspath(self.raw_output)
        else:
            outpath = self.path_input
        return outpath

    def save_output(self, fname_in, coord_list):
        """Save coordinate list to file with appropriate header."""
        is_radius = self.radius is not None
        if self.type == "txt":
            delimeter = " "
            header = "r c i" if is_radius else "r c"
        else:
            delimeter = ","
            header = "r,c,i" if is_radius else "r c"

        fname_out = os.path.join(self.path_output, f"{basename(fname_in)}.{self.type}")
        np.savetxt(
            fname_out,
            coord_list,
            fmt="%.4f",
            delimiter=delimeter,
            header=header,
            comments="",
        )

    def predict_single(self, fname_in, model):
        """Predict and save a single image."""
        image = load_image(fname_in)
        coord_list = predict(image, model)

        if self.radius is not None:
            coord_list = get_intensities(image, coord_list, self.radius)

        self.save_output(fname_in, coord_list)
        if self.verbose:
            self.logger.log(20, f"\U0001F3C3 file {fname_in} prediction complete.")

    def run(self):
        """Run prediction for all given images."""
        model = load_model(
            os.path.abspath(self.fname_model)
        )  # noqa: assignment-from-no-return
        if self.verbose:
            self.logger.log(20, "\U0001F9E0 model imported.")
            self.logger.log(20, f"\U0001F4C2 {len(self.file_list)} file(s) found.")
            self.logger.log(
                20, f"\U0001F5C4 output will be saved to {self.path_output}."
            )

        for fname_in in self.file_list:
            self.predict_single(fname_in, model)

        if self.verbose:
            self.logger.log(20, "\U0001F3C1\U0001F3C3 all predictions are complete.")

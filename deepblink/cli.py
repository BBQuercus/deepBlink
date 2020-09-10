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
from typing import List
import argparse
import logging
import os
import sys
import textwrap
import yaml

import numpy as np
import skimage.io
import tensorflow as tf

from .inference import get_intensities
from .inference import predict
from .io import basename
from .io import grab_files
from .io import load_image
from .io import load_model
from .training import run_experiment

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


def _parse_args_train(
    subparsers: argparse._SubParsersAction,
) -> argparse._SubParsersAction:
    """Subparser for training."""
    parser = subparsers.add_parser(
        "train", help="\U0001F35E train a freshly baked model on a dataset",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the experimental config.yaml file. Check the GitHub repository for an example",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=None,
        help="index of GPU to be used [default: None]",
    )
    return subparsers


def _parse_args_check(
    subparsers: argparse._SubParsersAction,
) -> argparse._SubParsersAction:
    """Subparser for checking."""
    parser = subparsers.add_parser(
        "check", help="\U0001F537 \U0001F535 determine your input images' shape"
    )
    parser.add_argument(
        "INPUT", type=str, help=f"input image location [filetypes: {EXTENSIONS}]",
    )
    return subparsers


def _parse_args_predict(
    subparsers: argparse._SubParsersAction,
) -> argparse._SubParsersAction:
    """Subparser for prediction."""
    parser = subparsers.add_parser(
        "predict",
        help="\U0001F914 inference / prediction of data with a pre-trained model",
    )
    parser.add_argument(
        "MODEL", type=argparse.FileType("r"), help="model .h5 file location"
    )
    parser.add_argument(
        "INPUT",
        type=FileFolderType(),
        help=f"input file/folder location [filetypes: {EXTENSIONS}]",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=FolderType(),
        help="output file/folder location [default: input location]",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["csv", "txt"],
        default="csv",
        help="output file type [options: csv, txt] [default: csv]",
    )
    parser.add_argument(
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
    return subparsers


def _parse_args_eval(
    subparsers: argparse._SubParsersAction,
) -> argparse._SubParsersAction:
    """Subparser for evaluation."""
    parser = subparsers.add_parser(
        "eval", help="\U0001F3AD measure a models performance on a dataset"
    )
    parser.add_argument(
        "INPUT",
        type=FileFolderType(),
        help=f"input file/folder location [filetypes: {EXTENSIONS}]",
    )
    return subparsers


def _parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(
        prog="deepblink",
        description=textwrap.dedent(
            """deepBlinks command line interface \U0001F469\U0000200D\U0001F4BB
        for training, inferencing, and evaluation"""
        ),
        epilog="We hope you enjoy using deepBlink \U0001F603",
    )
    parser.add_argument("-V", "--version", action="version", version="%(prog)s 0.0.6")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="set program output to verbose [default: quiet]",
    )
    parser.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)

    subparsers = parser.add_subparsers(dest="command")
    subparsers = _parse_args_train(subparsers)
    subparsers = _parse_args_check(subparsers)
    subparsers = _parse_args_predict(subparsers)
    subparsers = _parse_args_eval(subparsers)

    args = parser.parse_args()
    return args


def _configure_logger(verbose: bool, debug: bool):
    """Return verbose logger with three levels.

    * Verbose false and debug false - no verbose logging.
    * Verbose true and debug false - only info level loginfo for standard users.
    * Debug true - debug mode for developers.
    """
    if debug:
        level = logging.DEBUG
    else:
        if verbose:
            level = logging.INFO
        else:
            level = logging.CRITICAL

    logging.basicConfig(
        format="%(asctime)s: %(message)s", stream=sys.stdout, level=level
    )
    logger = logging.getLogger("Verbose output logger")
    return logger


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
        return config

    def set_gpu(self):
        """Set GPU environment variable."""
        if self.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{self.gpu}"

    def run(self):
        """Set configuration and start training loop."""
        self.set_gpu()
        run_experiment(self.config)


class HandleCheck:
    """Handle checking submodule for CLI.

    Args:
        arg_input: Path to image.
        logger: Verbose logger.
    """

    def __init__(self, arg_input: str, logger: logging.Logger):
        self.raw_input = arg_input
        self.logger = logger

        self.extensions = tuple(EXTENSIONS)
        self.abs_input = os.path.abspath(arg_input)
        self.logger.log(20, "\U0001F537 \U0001F535 starting checking submodule")

    # TODO merge with pink.io.load_image
    @property
    def image(self):
        """Load a single image."""
        if not os.path.isfile(self.abs_input):
            raise ImportError(
                "\U0000274C Input file does not exist. Please provide a valid path."
            )
        if not self.abs_input.lower().endswith(self.extensions):
            raise ImportError(
                f"\U0000274C Input file extension invalid. Please use one of {self.extensions}."
            )
        image = skimage.io.imread(self.abs_input).squeeze()
        return image

    @staticmethod
    def predict_shape(shape) -> str:
        """Predict the channel-arangement based on common standards."""
        is_rgb = 3 in shape
        max_len = 5 if is_rgb else 4
        if not any([len(shape) == i for i in range(2, max_len)]):
            raise ValueError("Shape can't be predicted.")

        dims = {}
        dims["x"], dims["y"] = [
            idx for idx, i in enumerate(shape) if i in sorted(shape)[-2:]
        ]
        sorted_shape = sorted(shape)
        if is_rgb:
            dims["3"] = shape.index(3)
            sorted_shape.remove(3)
        if len(sorted_shape) >= 3:
            dims["z"] = shape.index(sorted_shape[0])
        if len(sorted_shape) >= 4:
            dims["t"] = shape.index(sorted_shape[1])

        sorted_dims = [k for k, v in sorted(dims.items(), key=lambda item: item[1])]
        order = ",".join(sorted_dims)
        return order

    def run(self) -> None:
        """Run check for input image."""
        print(
            textwrap.dedent(
                f"""
        1. Your image has a shape of: {self.image.shape}
        ----------
        2. Possible parameters
        \U000027A1 x, y: single 2D image used for one prediction
        \U000027A1 z: third (height) dimension
        \U000027A1 t: time dimension
        \U000027A1 3: RGB color stack
        ----------
        3. By default we would assign: "({self.predict_shape(self.image.shape)})"
        \U0001F449 If this is incorrect, please provide the proper shape using the --shape flag to the
        submodule predict in deepblink's command line interface
        """
            )
        )


class HandlePredict:
    """Handle prediction submodule for CLI.

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

        self.logger.log(20, "\U0001F914 starting prediction submodule")

    @property
    def path_input(self) -> str:
        """Return absolute input path (dependent on file/folder input)."""
        if os.path.isdir(self.abs_input):
            path_input = self.abs_input
        elif os.path.isfile(self.abs_input):
            path_input = os.path.dirname(self.abs_input)
        return path_input

    @property
    def file_list(self) -> List[str]:
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
    def path_output(self) -> str:
        """Return the absolute output path (dependent if given)."""
        if os.path.exists(str(self.raw_output)):
            outpath = os.path.abspath(self.raw_output)
        else:
            outpath = self.path_input
        return outpath

    def save_output(self, fname_in: str, coord_list: np.ndarray) -> None:
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

    def predict_single(self, fname_in: str, model: tf.keras.models.Model) -> None:
        """Predict and save a single image."""
        image = load_image(fname_in)
        coord_list = predict(image, model)

        if self.radius is not None:
            coord_list = get_intensities(image, coord_list, self.radius)

        self.save_output(fname_in, coord_list)
        self.logger.log(20, f"\U0001F3C3 prediction of file {fname_in} complete")

    def run(self):
        """Run prediction for all given images."""
        model = load_model(
            os.path.abspath(self.fname_model)
        )  # noqa: assignment-from-no-return
        self.logger.log(20, "\U0001F9E0 model imported")
        self.logger.log(20, f"\U0001F4C2 {len(self.file_list)} file(s) found")
        self.logger.log(
            20, f"\U0001F5C4{' '} output will be saved to {self.path_output}"
        )

        for fname_in in self.file_list:
            self.predict_single(fname_in, model)

        self.logger.log(20, "\U0001F3C1\U0001F3C3 all predictions are complete")


def main():
    """Entrypoint for the CLI."""
    args = _parse_args()
    logger = _configure_logger(args.verbose, args.debug)

    if args.command == "train":
        handler = HandleTrain(arg_config=args.config, arg_gpu=args.gpu, logger=logger)

    if args.command == "check":
        handler = HandleCheck(arg_input=args.INPUT, logger=logger)

    if args.command == "predict":
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

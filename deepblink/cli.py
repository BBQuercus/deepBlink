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

Why does this file exist, and why not put this in __main__?
    - When you run `python -mdeepblink` or `deepblink` directly, python will execute ``__main__.py``
      as a script. That means there won't be any ``deepblink.__main__`` in ``sys.modules``.
    - When you import __main__ it will get executed again (as a module) because there's no
      ``deepblink.__main__`` in ``sys.modules``.
    - Therefore, to avoid double excecution of the code, this split-up way is safer.
"""
import argparse
import os
import textwrap

import numpy as np

from .inference import predict
from .inference import get_intensities
from .io import basename
from .io import grab_files
from .io import load_image
from .io import load_model

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
        description="Deepblink CLI for inferencing of new data with pretrained models.",
    )

    # Positional arguments
    parser.add_argument(
        "MODEL", type=argparse.FileType("r"), help="model .h5 file location"
    )
    parser.add_argument(
        "INPUT",
        type=FileFolderType(),
        help=f"input file/folder location [filetypes: {EXTENSIONS}]",
    )

    # Optional arguments
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
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="set program output to verbose [default: quiet]",
    )
    parser.add_argument("-V", "--version", action="version", version="%(prog)s 0.0.6")
    args = parser.parse_args()

    return args


def main():
    """Entrypoint for the CLI."""
    args = _parse_args()

    # Model import
    fname_model = os.path.abspath(args.MODEL.name)
    model = load_model(fname_model)  # noqa: assignment-from-no-return
    if args.verbose:
        print("Model imported.")

    # File listing
    inputs = os.path.abspath(args.INPUT)
    if os.path.isdir(inputs):
        files = grab_files(inputs, EXTENSIONS)
        inpath = inputs
    elif os.path.isfile(inputs):
        files = [inputs]
        inpath = os.path.dirname(inputs)
    else:
        raise ImportError("Input file(s) could not be found.")
    if args.verbose:
        print(f"{len(files)} file(s) found.")

    # Output path definition
    if os.path.exists(str(args.output)):
        outpath = os.path.abspath(args.output)
    else:
        outpath = inpath

    # Header definition
    delimeter = " " if args.type == "txt" else ","
    if args.radius is not None:
        header = "r c i" if args.type == "txt" else "r,c,i"
    else:
        header = "r c" if args.type == "txt" else "r,c"

    for file in files:
        # Image import
        image = load_image(file)

        # Prediction
        coord = predict(image, model)

        # Optional intensity calculation
        if args.radius is not None:
            coord = get_intensities(image, coord, args.radius)

        # Save coord list
        fname = os.path.join(outpath, f"{basename(file)}.{args.type}")
        np.savetxt(
            fname, coord, fmt="%.4f", delimiter=delimeter, header=header, comments=""
        )
    if args.verbose:
        print("Predictions complete.")

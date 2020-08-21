"""Module that contains the command line app.

Used for inferencing of new data with pretrained models.
Because the model is loaded into memory for every run, it is faster to
run multiple images at once by passing a directory as input.

Usage:
    ``$ deepblink [-h] [-o OUTPUT] [-t {csv,txt}] [-v] [-V] MODEL INPUT``
    ``$ deepblink --help``

Positional Arguments:
    MODEL           model .h5 file location
    INPUT           input file/folder location

Optional Arguments:
    -h, --help      show this help screen

    -o, --output    output file/folder location [default: input location]
    -t, --type      output file type [options: csv, txt] [default: csv]
    -v, --verbose   set program output to verbose [default: quiet]
    -V, --version   show version program's version number and exit

Why does this file exist, and why not put this in __main__?
    - When you run `python -mdeepblink` or `deepblink` directly, python will execute ``__main__.py``
      as a script. That means there won't be any ``deepblink.__main__`` in ``sys.modules``.
    - When you import __main__ it will get executed again (as a module) because there's no
      ``deepblink.__main__`` in ``sys.modules``.
    - Therefore, to avoid double excecution of the code, this split-up way is safer.
"""
from typing import List, Tuple
import argparse
import glob
import os

import numpy as np
import tensorflow as tf

from .data import get_coordinate_list
from .data import next_power
from .data import normalize_image
from .io import extract_basename
from .io import load_image
from .losses import combined_f1_rsme
from .losses import f1_score
from .losses import rmse

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
        type=str,
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
        "-v",
        "--verbose",
        action="store_true",
        help="output file/folder location [default: input location]",
    )
    parser.add_argument("-V", "--version", action="version", version="%(prog)s 0.0.5")
    args = parser.parse_args()

    return args


def _grab_files(path: str, extensions: List[str]) -> List[str]:
    """Grab all files in directory with listed extensions.

    Args:
        path: Path to files to be grabbed. Without trailing "/".
        extensions: List of all file extensions. Without leading ".".

    Returns:
        Sorted list of all corresponding files.

    Raises:
        OSError: Path not existing.
    """
    if not os.path.exists(path):
        raise OSError(f"Path must exist. '{path}' does not.")

    files = []
    for ext in extensions:
        files.extend(glob.glob(f"{path}/*.{ext}"))
    return sorted(files)


def _predict(
    image: np.ndarray, model: tf.keras.models.Model
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a binary or categorical model based prediction of an image.

    Args:
        image: Image to be predicted.
        model: Model used to predict the image.

    Returns:
        List of coordinates [r, c].
    """
    # Normalisation and padding
    image = normalize_image(image)
    pad_bottom = next_power(image.shape[0], 2) - image.shape[0]
    pad_right = next_power(image.shape[1], 2) - image.shape[1]
    image = np.pad(image, ((0, pad_bottom), (0, pad_right)), "reflect")

    # Predict on image
    pred = model.predict(image[None, ..., None]).squeeze()
    coords = get_coordinate_list(pred, image.shape[0])

    # Remove spots in padded part of image
    coords = np.array([coords[..., 0], coords[..., 1]])
    coords = np.where(
        (coords[0] < image.shape[1]) & (coords[1] < image.shape[0]), coords, None
    )

    return coords.T  # Transposition to save as rows


def main():
    """Entrypoint for the CLI."""
    args = _parse_args()

    # Model import
    model = os.path.abspath(args.MODEL.name)
    model = tf.keras.models.load_model(
        model,
        custom_objects={
            "f1_score": f1_score,
            "rmse": rmse,
            "combined_f1_rsme": combined_f1_rsme,
        },
    )
    if args.verbose:
        print("Model imported.")

    # File listing
    inputs = os.path.abspath(args.INPUT)
    if os.path.isdir(inputs):
        files = _grab_files(inputs, EXTENSIONS)
        inpath = inputs
    elif os.path.isfile(inputs):
        files = [inputs]
        inpath = os.path.dirname(inputs)
    else:
        raise ImportError("Input file(s) could not be found.")
    if args.verbose:
        print(f"{len(files)} file(s) found.")

    # Output path definition
    if args.output is not None and os.path.exists(args.output):
        outpath = os.path.abspath(args.output)
    else:
        outpath = inpath
    delimeter = " " if args.type == "txt" else ","
    header = "r c" if args.type == "txt" else "r,c"

    for file in files:
        # Image import
        image = load_image(file)

        # Prediction
        coord = _predict(image, model)

        # Save coord list
        fname = os.path.join(outpath, f"{extract_basename(file)}.{args.type}")
        np.savetxt(
            fname, coord, fmt="%.4f", delimiter=delimeter, header=header, comments=""
        )
    if args.verbose:
        print("Predictions complete.")

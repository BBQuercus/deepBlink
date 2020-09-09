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
from typing import List
import argparse
import glob
import os
import textwrap

import numpy as np
import skimage.morphology
import tensorflow as tf

from .data import get_coordinate_list
from .data import next_power
from .data import normalize_image
from .io import basename
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


def _predict(image: np.ndarray, model: tf.keras.models.Model) -> np.ndarray:
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
    image_pad = np.pad(image, ((0, pad_bottom), (0, pad_right)), "reflect")

    # Predict on image
    pred = model.predict(image_pad[None, ..., None]).squeeze()
    coords = get_coordinate_list(pred, image_pad.shape[0])

    # Remove spots in padded part of image
    coords = np.array([coords[..., 0], coords[..., 1]])
    coords = np.delete(
        coords,
        np.where((coords[0] > image.shape[0]) | (coords[1] > image.shape[1])),
        axis=1,
    )

    return coords.T  # Transposition to save as rows


def get_intensities(
    image: np.ndarray, coordinate_list: np.ndarray, radius: int
) -> np.ndarray:
    """Finds integrated intensities in a radius around each coordinate.

    Args:
        image: Input image with pixel values.
        coordinate_list: List of r, c coordinates in shape (n, 2).
        radius: Radius of kernel to determine intensities.

    Returns:
        Array with all integrated intensities.
    """
    kernel = skimage.morphology.disk(radius)

    for r, c in coordinate_list:
        if not all([isinstance(i, float) for i in [r, c]]):
            print(r, c)

    intensities = np.zeros((len(coordinate_list), 1))
    for idx, (r, c) in enumerate(np.round(coordinate_list).astype(int)):
        # Selection with indexes will be truncated to the max index possible automatically
        area = (
            image[
                max(r - radius, 0) : r + radius + 1,
                max(c - radius, 0) : c + radius + 1,
            ]
            * kernel[
                max(radius - r, 0) : radius + image.shape[0] - r,
                max(radius - c, 0) : radius + image.shape[1] - c,
            ]
        )
        intensities[idx] = np.sum(area)

    output = np.append(coordinate_list, intensities, axis=1)
    return output


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
        coord = _predict(image, model)

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

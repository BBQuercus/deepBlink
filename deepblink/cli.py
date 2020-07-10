"""Module that contains the command line app.

Used for inferencing of new data with pretrained models.
Because the model is loaded into memory for every run, it is faster to
run multiple images at once by passing a directory as input.

Usage:
    ``$ deepblink MODEL INPUT [-o, --output] OUTPUT [-t, --type] TYPE [-v, --verbose]``
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
import argparse
import glob
import math
import os
from typing import List
from typing import Tuple

import numpy as np
import skimage.color
import skimage.io
import tensorflow as tf

from .data import get_coordinate_list
from .data import next_multiple
from .io import extract_basename
from .losses import f1_l2_combined_loss
from .losses import f1_score
from .losses import l2_norm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
"""Removes tensorflow's information on CPU / GPU availablity."""

EXTENSIONS = ["tif", "jpeg", "jpg", "png"]
"""List[str]: List of currently supported image file extensions."""


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
        type=argparse.FileType("r"),
        help=f"input file/folder location [filetypes: {EXTENSIONS}]",
    )

    # Optional arguments
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
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
    parser.add_argument("-V", "--version", action="version", version="%(prog)s 0.0.4")
    args = parser.parse_args()

    return args


def _grab_files(path: str, extensions: List[str]):
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


def _import_image(fname: str):
    """Import a single image as numpy array checking format requirements."""
    try:
        image = skimage.io.imread(fname).squeeze()
        if image.ndim == 3 and image.shape[2] == 3:
            return skimage.color.rgb2gray(image)
        if image.ndim == 2 and image.shape[0] > 0 and image.shape[1] > 0:
            return image
        raise ValueError(
            f"File must be in the format (x, y) or (x, y, 3) but is {image.shape}."
        )
    except ValueError:
        raise ImportError(f"File '{fname}' could not be imported.")


def _predict(image: np.ndarray, model: tf.keras.models.Model) -> np.ndarray:
    """Predict on a image of size needed for the network and return coordinates."""
    if not image.shape[0] == image.shape[1]:
        raise ValueError(f"Image shape must be square but is {image.shape}")
    if not math.log(image.shape[0], 2).is_integer():
        raise ValueError(f"Image shape must a power of two but is {image.shape}")
    if image.shape[0] != model.input_shape[1]:
        raise ValueError(
            f"Image shape must match model {image.shape} != {model.input_shape[1:3]}"
        )

    pred = model.predict(image[None, ..., None]).squeeze()
    coord = get_coordinate_list(pred, model.input_shape[1])
    return coord[..., 0], coord[..., 1]


def predict_baseline(
    image: np.ndarray, model: tf.keras.models.Model
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns a binary or categorical model based prediction of an image.

    Args:
        image: Image to be predicted.
        model: Model used to predict the image.

    Returns:
        List of coordinates [x,y].
    """
    input_size = model.layers[0].output_shape[0][1]

    # Normalisation and padding
    image /= np.max(image)
    pad_bottom = next_multiple(image.shape[0], input_size) - image.shape[0]
    pad_right = next_multiple(image.shape[1], input_size) - image.shape[1]
    image = np.pad(image, ((0, pad_bottom), (0, pad_right)), "reflect")

    # Predict on patches of the image and combine all the patches
    crops = skimage.util.view_as_windows(
        image, (input_size, input_size), step=input_size
    )
    coords_x = []
    coords_y = []

    for i in range(crops.shape[0]):
        for j in range(crops.shape[1]):
            x, y = _predict(crops[i, j], model)
            abs_coord_x = x + (j * input_size)
            abs_coord_y = y + (i * input_size)

            coords_x.extend(abs_coord_x)
            coords_y.extend(abs_coord_y)

    coords = np.array([coords_x, coords_y])
    coords = np.where(
        (coords[0] < image.shape[1]) & (coords[1] < image.shape[0]), coords, None
    )

    return coords.T  # Transposition to save as rows


# TODO add verbosity changes
# TODO rename functions
def main():
    """Entrypoint for the CLI."""
    args = _parse_args()

    # Model import
    model = os.path.abspath(args.MODEL.name)
    model = tf.keras.models.load_model(
        model,
        custom_objects={
            "f1_score": f1_score,
            "l2_norm": l2_norm,
            "f1_l2_combined_loss": f1_l2_combined_loss,
        },
    )

    # File listing
    inputs = os.path.abspath(args.INPUT.name)
    if os.path.isdir(inputs):
        files = _grab_files(inputs, EXTENSIONS)
        inpath = inputs
    elif os.path.isfile(inputs):
        files = [inputs]
        inpath = os.path.dirname(inputs)
    else:
        raise ImportError("Input file(s) could not be found.")

    # File import
    images = [_import_image(file) for file in files]

    # Prediction
    coords = [predict_baseline(image, model) for image in images]

    # Save coord list
    if args.output is not None and os.path.exists(args.output):
        outpath = os.path.abspath(args.output)
    else:
        outpath = inpath
    delimeter = " " if args.type == "txt" else ","
    header = "x y" if args.type == "txt" else "x,y"

    for file, coord in zip(files, coords):
        fname = os.path.join(outpath, f"{extract_basename(file)}.{args.type}")
        np.savetxt(
            fname, coord, fmt="%.4f", delimiter=delimeter, header=header, comments=""
        )

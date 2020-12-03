"""Dataset preparation functions."""

from typing import Any, List, Tuple
import glob
import os
import re

import numpy as np
import skimage.color
import skimage.io
import tensorflow as tf

from .losses import combined_bce_rmse
from .losses import combined_dice_rmse
from .losses import combined_f1_rmse
from .losses import f1_score
from .losses import rmse

# List of currently supported image file extensions.
EXTENSIONS = ("tif", "tiff", "jpeg", "jpg", "png")


def basename(path: str) -> str:
    """Returns the basename removing path and extension."""
    return os.path.splitext(os.path.basename(path))[0]


def securename(fname: str) -> str:
    """Turns potentially unsafe names into a single, safe, alphanumeric string."""
    return re.sub(r"[^\w\d-]", "_", fname)


def load_npz(fname: str, test_only: bool = False) -> List[Any]:
    """Imports the standard npz file format used for custom training and inference.

    Only for files saved using "np.savez_compressed(fname, x_train, y_train...)".

    Args:
        fname: Path to npz file.
        test_only: Only return testing images and labels.

    Returns:
        A list of the required numpy arrays. If no "test_only" arguments were passed,
        returns [x_train, y_train, x_valid, y_valid, x_test, y_test].

    Raises:
        ValueError: If not all datasets are found.
    """
    expected = ["x_train", "y_train", "x_valid", "y_valid", "x_test", "y_test"]
    if test_only:
        expected = expected[-2:]

    with np.load(fname, allow_pickle=True) as data:
        if not all([e in data.files for e in expected]):
            raise ValueError(f"{expected} must be present. Only found {data.files}.")
        return [data[f] for f in expected]


def load_image(
    fname: str, extensions: Tuple[str, ...] = EXTENSIONS, is_rgb: bool = False
) -> np.ndarray:
    """Import a single image as numpy array checking format requirements.

    Args:
        fname: Absolute or relative filepath of image.
        extensions: Allowed image extensions.
        is_rgb: If true, converts RGB images to grayscale.
    """
    if not os.path.isfile(fname):
        raise ImportError("Input file does not exist. Please provide a valid path.")
    if not fname.lower().endswith(extensions):
        raise ImportError(f"Input file extension invalid. Please use {extensions}.")
    try:
        image = skimage.io.imread(fname).squeeze().astype(np.float32)
    except ValueError as error:
        raise ImportError(f"File '{fname}' could not be imported.") from error
    if is_rgb:
        image = skimage.color.rgb2gray(image)
    return image


def load_model(fname: str) -> tf.keras.models.Model:
    """Import a deepBlink model from file."""
    if not os.path.isfile(fname):
        raise ValueError(f"File must exist - '{fname}' does not.")
    if os.path.splitext(fname)[-1] != ".h5":
        raise ValueError(f"File must be of type h5 - '{fname}' does not.")

    try:
        model = tf.keras.models.load_model(
            fname,
            custom_objects={
                "combined_bce_rmse": combined_bce_rmse,
                "combined_dice_rmse": combined_dice_rmse,
                "combined_f1_rmse": combined_f1_rmse,
                "f1_score": f1_score,
                "leaky_relu": tf.nn.leaky_relu,
                "rmse": rmse,
            },
        )
        return model
    except ValueError as error:
        raise ImportError(f"Model '{fname}' could not be imported.") from error


def grab_files(path: str, extensions: Tuple[str, ...]) -> List[str]:
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
        files.extend(sorted(glob.glob(f"{path}/*.{ext}")))
    return sorted(files)

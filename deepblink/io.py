"""Dataset preparation functions."""

import os
from typing import Tuple

import numpy as np
import skimage.color
import skimage.io


def remove_zeros(lst: list) -> list:
    """Removes all occurences of "0" from a list of numpy arrays."""
    return [i for i in lst if isinstance(i, np.ndarray)]


# TODO rename to "basename"
def extract_basename(path: str) -> str:
    """Returns the basename removing path and extension."""
    return os.path.splitext(os.path.basename(path))[0]


# TODO check if files have x_train etc.
def load_npz(fname: str,) -> Tuple[np.ndarray, ...]:
    """Imports the standard npz file format used for custom training and inference.

    Only for files saved using "np.savez_compressed(fname, x_train, y_train...)".

    Args:
        fname: Path to npz file.

    Returns:
        (x_train, y_train, x_valid, y_valid, x_test, y_test) as numpy arrays.
    """
    with np.load(fname, allow_pickle=True) as data:
        return (
            data["x_train"],
            data["y_train"],
            data["x_valid"],
            data["y_valid"],
            data["x_test"],
            data["y_test"],
        )


def load_image(fname: str):
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

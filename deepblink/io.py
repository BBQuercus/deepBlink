"""Dataset preparation functions."""

from typing import Any, List
import os
import warnings

import numpy as np
import skimage.color
import skimage.io
import tensorflow as tf

from .losses import combined_bce_rmse
from .losses import combined_f1_rmse
from .losses import f1_score
from .losses import rmse


def remove_zeros(lst: list) -> list:
    """Removes all occurences of "0" from a list of numpy arrays."""
    return [i for i in lst if isinstance(i, np.ndarray)]


def extract_basename(path: str) -> str:
    """Depreciated name: Returns the basename removing path and extension."""
    warnings.warn(
        'Will be renamed to "basename" in the next release.', DeprecationWarning
    )
    return basename(path)


def basename(path: str) -> str:
    """Returns the basename removing path and extension."""
    return os.path.splitext(os.path.basename(path))[0]


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


def load_image(fname: str) -> np.ndarray:
    """Import a single image as numpy array checking format requirements."""
    try:
        image = skimage.io.imread(fname).squeeze().astype(np.float32)
        if image.ndim == 3 and image.shape[2] == 3:
            return skimage.color.rgb2gray(image)
        if image.ndim == 2 and image.shape[0] > 0 and image.shape[1] > 0:
            return image
        raise ValueError(
            f"File must be in the format (x, y) or (x, y, 3) but is {image.shape}."
        )
    except ValueError as error:
        raise ImportError(f"File '{fname}' could not be imported.") from error


def load_model(fname: str) -> tf.keras.models.Model:
    """Import a deepBlink model from file."""
    if not os.path.isfile(fname):
        raise ValueError(f"File must exist - '{fname}' does not.")
    if os.path.splitext(fname)[-1] != "h5":
        raise ValueError(f"File must be of type h5 - '{fname}' does not.")

    try:
        model = tf.keras.models.load_model(
            fname,
            custom_objects={
                "f1_score": f1_score,
                "rmse": rmse,
                "combined_bce_rmse": combined_bce_rmse,
                "combined_f1_rmse": combined_f1_rmse,
            },
        )
        return model
    except ValueError as error:
        raise ImportError(f"Model '{fname}' could not be imported.") from error

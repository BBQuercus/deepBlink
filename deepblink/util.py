"""Utility helper functions."""

from typing import Callable, Iterable, Tuple, Union
import importlib
import os
import random

from PIL import Image
from PIL.TiffTags import TAGS
import numpy as np
import pandas as pd


def get_from_module(path: str, attribute: str) -> Callable:
    """Grab an attribute (e.g. class) from a given module path."""
    module = importlib.import_module(path)
    attribute = getattr(module, attribute)
    return attribute  # type: ignore[return-value]


def relative_shuffle(
    x: Union[list, np.ndarray], y: Union[list, np.ndarray]
) -> Tuple[Union[list, np.ndarray], Union[list, np.ndarray]]:
    """Shuffles x and y keeping their relative order."""
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        shuffled_indices = np.random.permutation(x.shape[0])
        shuffled_x, shuffled_y = x[shuffled_indices], y[shuffled_indices]
    else:
        combined_list = list(zip(x, y))
        random.shuffle(combined_list)
        shuffled_x, shuffled_y = zip(*combined_list)
        shuffled_x, shuffled_y = list(shuffled_x), list(shuffled_y)
    return shuffled_x, shuffled_y


def train_valid_split(
    x_list: list, y_list: list, valid_split: float = 0.2, shuffle: bool = True
) -> Iterable[list]:
    """Split two lists (usually input and ground truth).

    Splitting into random training and validation sets with an optional shuffling.

    Args:
        x_list: First list of items. Typically input data.
        y_list: Second list of items. Typically labeled data.
        valid_split: Number between 0-1 to denote the percentage of examples used for validation.

    Returns:
        (x_train, x_valid, y_train, y_valid) splited lists containing training
        or validation examples respectively.
    """
    if not 0 <= valid_split <= 1:
        raise ValueError(f"valid_split must be between 0-1 but is {valid_split}.")
    if len(x_list) != len(y_list):
        raise ValueError(
            f"Lists must be of equal length: {len(x_list)} != {len(y_list)}."
        )
    if len(x_list) <= 2:
        raise ValueError(
            f"At least 3 images/labels are required for a train/val split. Given: {len(x_list)}."
        )

    if shuffle:
        x_list, y_list = relative_shuffle(x_list, y_list)

    split_len = round(len(x_list) * valid_split)

    x_valid = x_list[:split_len]
    x_train = x_list[split_len:]
    y_valid = y_list[:split_len]
    y_train = y_list[split_len:]

    return x_train, x_valid, y_train, y_valid


def delete_non_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Deletes DataFrame columns that only contain one (non-unique) value."""
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col, inplace=True, axis=1)
    return df


def remove_falses(tup: tuple) -> tuple:
    """Removes all false occurences from a tuple."""
    return tuple([i for i in tup if i])


def predict_shape(shape: tuple) -> str:
    """Predict the channel-arangement based on common standards.

    Assumes the following things:
    * x, y are the two largest axes
    * rgb only if the last axis is 3
    * up to 4 channels
    * "fill up order" is c, z, t

    Args:
        shape: To be predicted shape. Output from np.ndarray.shape
    """
    shape = remove_falses(shape)
    if len(shape) < 2:
        raise ValueError(f"Too few dimensions. Shape {shape} can't be predicted.")

    is_rgb = shape[-1] == 3
    is_channel = any(i in shape for i in range(5)) and not is_rgb
    max_len = 6 if is_rgb else 5

    if len(shape) > max_len:
        raise ValueError(
            f"Too many dimensions (max {max_len}). Shape {shape} can't be predicted."
        )

    dims = {}
    dims["x"], dims["y"] = [
        idx for idx, i in enumerate(shape) if i in sorted(shape)[-2:]
    ]
    sorted_shape = sorted(shape)

    if is_rgb:
        dims["3"] = len(shape)
        sorted_shape.remove(3)
    if is_channel:
        dims["c"] = shape.index(sorted_shape.pop(0))
    if len(sorted_shape) >= 3:
        dims["z"] = shape.index(sorted_shape.pop(0))
    if len(sorted_shape) == 3:
        dims["t"] = shape.index(sorted_shape.pop(0))

    sorted_dims = [k for k, v in sorted(dims.items(), key=lambda item: item[1])]
    order = ",".join(sorted_dims)
    return order


# TODO account for resolution unit (µm / cm)
def predict_pixel_size(fname: Union[str, "os.PathLike[str]"]) -> Tuple[float, float]:
    """Predict the pixel size based on tifffile metadata."""
    if not os.path.splitext(fname)[1] == ".tif":
        raise ValueError(f"{fname} is not a tif file.")
    if not os.path.isfile(fname):
        raise ValueError(f"{fname} does not exist.")

    image = Image.open(fname)
    if len(image.size) != 2:
        raise ValueError(f"Image {fname} has more than 2 dimensions.")

    # Get resolutions from tiff metadata
    meta_dict = {TAGS[key]: image.tag[key] for key in image.tag_v2}
    x_res = meta_dict.get("XResolution", ((1, 1),))
    y_res = meta_dict.get("YResolution", ((1, 1),))
    unit = meta_dict.get("ResolutionUnit", (1,))
    x_size = x_res[0][1] / x_res[0][0] * unit[0]
    y_size = y_res[0][1] / y_res[0][0] * unit[0]

    return x_size, y_size

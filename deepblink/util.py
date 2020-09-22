"""Utility helper functions."""

from typing import Callable, Iterable, Tuple
import importlib
import random

import pandas as pd


def get_from_module(path: str, attribute: str) -> Callable:
    """Grab an attribute (e.g. class) from a given module path."""
    module = importlib.import_module(path)
    attribute = getattr(module, attribute)
    return attribute  # type: ignore[return-value]


def relative_shuffle(x_list: list, y_list: list) -> Tuple[list, list]:
    """Shuffles two list keeping their relative arrangement."""
    combined = list(zip(x_list, y_list))
    random.shuffle(combined)
    x_tuple, y_tuple = zip(*combined)
    return list(x_tuple), list(y_tuple)


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

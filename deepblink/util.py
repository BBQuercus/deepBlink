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
        raise ValueError("Lists must contain 3 elements or more.")

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

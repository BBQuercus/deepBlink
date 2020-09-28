"""Baseline Dataset class."""

import os


class Dataset:
    """Simple abstract class for datasets.

    Args:
        name: Absolute path to dataset file.
    """

    def __init__(self, name: str, *_):
        self.name = name if name else None
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.x_test = None
        self.y_test = None

    @property
    def data_filename(self) -> str:  # type: ignore[return-value]
        """Return the absolute path to the dataset."""
        return os.path.abspath(self.name)  # type: ignore

    def load_data(self):
        """Empty method to import or create data."""

    def prepare_data(self):
        """Empty method to prepare or convert data."""

    def normalize_dataset(self):
        """Empty method to normalise images in the dataset."""

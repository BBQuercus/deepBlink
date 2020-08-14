"""Baseline Dataset class."""

import pathlib


class Dataset:
    """Simple abstract class for datasets."""

    def __init__(self, name: str, *_):
        self.name = name if name else None
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.x_test = None
        self.y_test = None

    @classmethod
    def data_dirname(cls):
        """Return the absolute path to data files."""
        return pathlib.Path(__file__).resolve().parents[2] / "data"

    def load_data(self):
        """Empty method to import or create data."""

    def prepare_data(self):
        """Empty method to prepare or convert data."""

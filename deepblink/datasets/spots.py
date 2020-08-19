"""SpotsDataset class."""

import os

import numpy as np

from ..data import get_prediction_matrix
from ..io import load_npz
from ._datasets import Dataset

DATA_DIRNAME = Dataset.data_dirname()


class SpotsDataset(Dataset):
    """Class used to load all spots data.

    Args:
        cell_size: Number of pixels (from original image) constituting
            one cell in the prediction matrix.
    """

    def __init__(self, name: str, cell_size: int):
        super().__init__(name)
        self.cell_size = cell_size

    @property
    def data_filename(self) -> str:  # type: ignore[return-value]
        """Return the absolute path to the dataset."""
        return os.path.abspath(self.name)  # type: ignore

    def load_data(self) -> None:
        """Load dataset into memory."""
        (
            self.x_train,
            self.y_train,
            self.x_valid,
            self.y_valid,
            self.x_test,
            self.y_test,
        ) = load_npz(self.data_filename)
        self.prepare_data()

    def prepare_data(self) -> None:
        """Convert raw labels into labels usable for training.

        In the "spots" format, training labels are stored as lists of coordinates,
        this format cannot be used for training. Here, this format is converted into
        prediction matrices.
        """
        image_size = self.x_train[0].shape[0]  # type: ignore

        def __conversion(dataset, image_size, cell_size):
            return np.array(
                [
                    get_prediction_matrix(coords, image_size, cell_size)
                    for coords in dataset
                ]
            )

        self.y_train = __conversion(self.y_train, image_size, self.cell_size)
        self.y_valid = __conversion(self.y_valid, image_size, self.cell_size)
        self.y_test = __conversion(self.y_test, image_size, self.cell_size)

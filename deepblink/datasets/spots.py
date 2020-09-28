"""SpotsDataset class."""

import numpy as np

from ..data import get_prediction_matrix
from ..data import next_power
from ..data import normalize_image
from ..io import load_npz
from ._datasets import Dataset


class SpotsDataset(Dataset):
    """Class used to load all spots data.

    Args:
        cell_size: Number of pixels (from original image) constituting
            one cell in the prediction matrix.
    """

    def __init__(self, name: str, cell_size: int):
        super().__init__(name)
        self.cell_size = cell_size
        self.load_data()

    def load_data(self) -> None:
        """Load dataset into memory."""
        self.x_train, self.y_train, self.x_valid, self.y_valid, _, _ = load_npz(
            self.data_filename
        )
        self.prepare_data()
        self.normalize_dataset()

    @property
    def image_size(self):
        """Check if all images have the same square shape."""
        base_shape = self.x_train[0].shape
        if not all(
            base_shape == x.shape
            for x in dataset
            for dataset in [self.x_train, self.x_valid]
        ):
            raise ValueError("All images must have the same shape.")
        if not base_shape[0] == base_shape[1]:
            raise ValueError("Images must be square. ")
        if not base_shape[0] == next_power(base_shape[0]):
            raise ValueError(
                f"Images sidelength must be a power of two. {base_shape[0]} is not."
            )

        return base_shape[0]

    def prepare_data(self) -> None:
        """Convert raw labels into labels usable for training.

        In the "spots" format, training labels are stored as lists of coordinates,
        this format cannot be used for training. Here, this format is converted into
        prediction matrices.
        """

        def __convert(dataset, image_size, cell_size):
            return np.array(
                [
                    get_prediction_matrix(coords, image_size, cell_size)
                    for coords in dataset
                ]
            )

        self.y_train = __convert(self.y_train, self.image_size, self.cell_size)
        self.y_valid = __convert(self.y_valid, self.image_size, self.cell_size)

    def normalize_dataset(self) -> None:
        """Normalize all the images to have zero mean and standard deviation 1."""

        def __normalize(dataset):
            return np.array([normalize_image(image) for image in dataset])

        self.x_train = __normalize(self.x_train)
        self.x_valid = __normalize(self.x_valid)

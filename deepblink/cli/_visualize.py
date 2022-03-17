"""CLI submodule for visualization."""

import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np

from ..io import load_image
from ..io import load_npz
from ..io import load_prediction


class HandleVisualize:
    """Handle checking submodule for CLI.

    Args:
        arg_config: Path to config.yaml file.
        arg_gpu: Which gpu is to be used.
        logger: Verbose logger.
    """

    def __init__(
        self,
        arg_dataset: str,
        arg_subset: str,
        arg_index: int,
        arg_image: str,
        arg_prediction: str,
        logger: logging.Logger,
    ):
        self.dataset = arg_dataset
        self.dataset_index = arg_index
        self.dataset_subset = arg_subset
        self.image = arg_image
        self.prediction = arg_prediction
        self.logger = logger
        self.logger.info("\U0001F4F8 starting visualization submodule")

    def __call__(self):
        """Select dataset / image and run visualization."""
        if self.dataset is not None:
            self.logger.info("\U0001F5BC Visualizing dataset provided.")
            self.visualize_dataset()
        elif self.image is not None:
            self.logger.info("\U0001F5BC visualizing image provided.")
            self.visualize_image()
        else:
            self.logger.error(
                "\U0000274C No input given. Please provide a dataset or image/prediction."
            )

    def visualize_dataset(self):
        """Visualize a random image and label from a npz file dataset."""
        # Load dataset and subset
        x_train, y_train, x_valid, y_valid, x_test, y_test = load_npz(self.dataset)
        if self.dataset_subset == "train":
            x = x_train
            y = y_train
        elif self.dataset_subset == "valid":
            x = x_valid
            y = y_valid
        elif self.dataset_subset == "test":
            x = x_test
            y = y_test
        else:
            self.logger.error(
                "Invalid dataset subset. Please select from train, valid, or test."
            )

        # Load image and prediction
        if self.dataset_index is not None and self.dataset_index >= len(x):
            self.logger.error(
                "Invalid dataset index. "  # nosec: B608
                f"Please select from 0 to end ({len(x) - 1})."
            )
        else:
            self.dataset_index = random.randint(0, len(x) - 1)  # nosec: B311
        image = x[self.dataset_index]
        coords = y[self.dataset_index]

        # Visualize
        self.plot_data(
            image,
            coords,
            f"Label (subset: {self.dataset_subset}, index: {self.dataset_index})",
        )

    def visualize_image(self):
        """Visualize an existing image and prediction (2D)."""
        image = load_image(self.image)
        if image.ndim != 2:
            self.logger.error("invalid image dimension")

        if self.prediction is None:
            self.prediction = f"{os.path.splitext(self.image)[0]}.csv"
            self.logger.debug("using same name prediction file")
        df = load_prediction(self.prediction)
        coords = df[["x [px]", "y [px]"]].values

        self.plot_data(image, coords, "Prediction")

    @staticmethod
    def plot_data(image: np.ndarray, coords: np.ndarray, label: str):
        """Plot image and prediction coordinates ([y, x])."""
        _, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
        ax[0].set_title("Raw Image")
        ax[0].imshow(1 - image, cmap="Blues")

        ax[1].set_title(label)
        ax[1].imshow(1 - image, cmap="Blues")
        ax[1].scatter(x=coords.T[1], y=coords.T[0], marker="x", color="r", s=4)
        plt.show()

"""SequenceDataset class."""

from typing import Callable
from typing import Tuple

import numpy as np
import tensorflow as tf


def _shuffle(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle x and y maintaining their association."""
    shuffled_indices = np.random.permutation(x.shape[0])
    return x[shuffled_indices], y[shuffled_indices]


class SequenceDataset(tf.keras.utils.Sequence):
    """Custom Sequence class used to feed data into model.fit.

    Args:
        x_list: List of inputs.
        y_list: List of targets.
        batch_size: Size of one mini-batch.
        augment_fn: Function to augment one mini-batch of x and y.
        format_fn: Function to format raw data to model input.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 16,
        augment_fn: Callable = None,
        format_fn: Callable = None,
    ):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.augment_fn = augment_fn
        self.format_fn = format_fn

    def __len__(self) -> int:
        """Return length of the dataset in unit of batch size."""
        if len(self.x) <= self.batch_size:
            print(
                "Warning! barch size larger than dataset, setting batch size to length of dataset"
            )
            self.batch_size = len(self.x)

        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        """Return a single batch."""
        # idx = 0  # Uncomment to overfit to just one batch
        begin = idx * self.batch_size
        end = (idx + 1) * self.batch_size

        batch_x = self.x[begin:end]
        batch_y = self.y[begin:end]

        if self.format_fn:
            batch_x, batch_y = self.format_fn(batch_x, batch_y)

        if self.augment_fn:
            batch_x, batch_y = self.augment_fn(batch_x, batch_y)

        if batch_x.ndim < 4:
            batch_x = np.expand_dims(batch_x, -1)
        if batch_y.ndim < 4:
            batch_y = np.expand_dims(batch_y, -1)

        return batch_x, batch_y

    def on_epoch_end(self) -> None:
        """Shuffle data after every epoch."""
        self.x, self.y = _shuffle(self.x, self.y)

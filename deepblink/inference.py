"""Model prediction / inference functions."""

from typing import Union
import math

import numpy as np
import skimage.morphology
import tensorflow as tf

from .data import get_coordinate_list
from .data import next_power
from .data import normalize_image


def predict(
    image: np.ndarray,
    model: tf.keras.models.Model,
    probability: Union[None, float] = None,
) -> np.ndarray:
    """Returns a binary or categorical model based prediction of an image.

    Args:
        image: Image to be predicted.
        model: Model used to predict the image.
        probability: Cutoff value to round model prediction probability.

    Returns:
        List of coordinates [r, c].
    """
    # Normalisation and padding
    image = normalize_image(image)
    pad_bottom = next_power(image.shape[0], 2) - image.shape[0]
    pad_right = next_power(image.shape[1], 2) - image.shape[1]
    image_pad = np.pad(image, ((0, pad_bottom), (0, pad_right)), "reflect")

    # Predict on image
    pred = model.predict(image_pad[None, ..., None]).squeeze()
    prob = 0.5 if probability is None else probability
    coords = get_coordinate_list(
        pred, image_size=max(image_pad.shape), probability=prob
    )

    # Remove spots in padded part of image
    coords = np.array([coords[..., 0], coords[..., 1]])
    coords = np.delete(
        coords,
        np.where((coords[0] > image.shape[0]) | (coords[1] > image.shape[1])),
        axis=1,
    )
    coords = coords.T  # Transposition to save as rows

    # Add third, probability containing column
    if probability is not None:
        probs = get_probabilities(pred, coords, image_size=max(image_pad.shape))
        probs = np.expand_dims(probs, axis=-1)
        coords = np.append(coords, probs, axis=-1)

    return coords


def get_probabilities(
    matrix: np.ndarray, coordinates: np.ndarray, image_size: int = 512
) -> np.ndarray:
    """Find prediction probability given the matrix and coordinates.

    Args:
        matrix: Matrix representation of spot coordinates.
        coordinates: Coordinates at which the probability should be determined.
        image_size: Default image size the grid was layed on.

    Returns:
        Array with all probabilities matching the coordinates.
    """
    matrix_size = max(matrix.shape)
    cell_size = image_size // matrix_size
    nrow = ncol = math.ceil(image_size / cell_size)

    probabilities = []
    for r, c in coordinates:
        # Position of cell coordinate in prediction matrix
        cell_r = min(nrow - 1, int(np.floor(r)) // cell_size)
        cell_c = min(ncol - 1, int(np.floor(c)) // cell_size)

        probabilities.append(matrix[cell_r, cell_c, 0])
    return np.array(probabilities)


def get_intensities(
    image: np.ndarray, coordinate_list: np.ndarray, radius: int
) -> np.ndarray:
    """Finds integrated intensities in a radius around each coordinate.

    Args:
        image: Input image with pixel values.
        coordinate_list: List of r, c coordinates in shape (n, 2).
        radius: Radius of kernel to determine intensities.

    Returns:
        Array with all integrated intensities.
    """
    kernel = skimage.morphology.disk(radius)

    for r, c in coordinate_list:
        if not all([isinstance(i, float) for i in [r, c]]):
            print(r, c)

    intensities = np.zeros((len(coordinate_list), 1))
    for idx, (r, c) in enumerate(np.round(coordinate_list).astype(int)):
        # Selection with indexes will be truncated to the max index possible automatically
        area = (
            image[
                max(r - radius, 0) : r + radius + 1,
                max(c - radius, 0) : c + radius + 1,
            ]
            * kernel[
                max(radius - r, 0) : radius + image.shape[0] - r,
                max(radius - c, 0) : radius + image.shape[1] - c,
            ]
        )
        intensities[idx] = np.sum(area)

    return intensities

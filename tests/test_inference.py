"""Unittests for the deepblink.inference module."""
# pylint: disable=missing-function-docstring,redefined-outer-name

import numpy as np
import pytest
import tensorflow as tf

from deepblink.inference import get_intensities
from deepblink.inference import predict
from deepblink.losses import combined_bce_rmse
from deepblink.losses import combined_f1_rmse
from deepblink.losses import f1_score
from deepblink.losses import rmse


def test_predict():
    """Test the function that given model and image, returns the prediction on image."""
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input((None, None, 1)),
            tf.keras.layers.Conv2D(3, 3, strides=2),
            tf.keras.layers.Activation("sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=combined_bce_rmse,
        metrics=[f1_score, rmse, combined_f1_rmse],
    )

    for size in [249, 512, 876]:
        image = np.random.rand(size, size)
        pred = predict(image, model)
        assert isinstance(pred, np.ndarray)


@pytest.fixture
def image():
    return np.ones((100, 100))


@pytest.fixture
def coordinates():
    return np.array([[0, 0], [20, 20], [0, 50]])


@pytest.mark.parametrize(
    "radius,expected", [(0, 3), (1, 3 + 5 + 4), (2, 6 + 13 + 9)],
)
def test_get_intensities(radius, expected, image, coordinates):
    output = get_intensities(image, coordinates, radius)
    output_sum = np.sum(output.T[2])  # Third column is intensity
    assert expected == output_sum

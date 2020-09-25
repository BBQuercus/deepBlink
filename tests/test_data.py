"""Unittests for the deepblink.data module."""
# pylint: disable=missing-function-docstring
# TODO random cropping

from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
import numpy as np
import pytest

from deepblink.data import next_power
from deepblink.data import next_multiple
from deepblink.data import normalize_image
from deepblink.data import get_coordinate_list
from deepblink.data import get_prediction_matrix


EPSILON = 1e-5


@pytest.mark.parametrize(
    "value, base, expected", [(64, 5, 2, 8), (2, 6, 3, 9), (64, 12, 7, 49)]
)
def test_next_power(value, base, expected):
    assert next_power(value, base) == expected


@pytest.mark.parametrize(
    "value, dividend, expected", [(5, 2, 6), (6, 3, 6), (8, 7, 14)]
)
def test_next_multiple(value, dividend, expected):
    assert next_multiple(value, dividend) == expected


@given(arrays(np.float, (5, 5), elements=floats(0, 100)))
def test_normalize_image(matrix):
    matrix = matrix + np.random.rand(5, 5)
    normalized_image = normalize_image(matrix)
    assert np.abs(np.mean(normalized_image)) < EPSILON
    assert np.abs(np.std(normalized_image) - 1) < EPSILON


def test_get_coordinate_list():
    grid_size = 4
    image_size = 512

    matrix = np.zeros((grid_size, grid_size, 3))
    matrix[0, 1, ...] = [1, 0.5, 0.5]

    theoretical_result = np.array(
        [image_size // grid_size * 0.5, image_size // grid_size * (1 + 0.5)]
    )
    output = get_coordinate_list(matrix, image_size=image_size)
    assert (theoretical_result == output).all()


def test_get_prediction_matrix():
    image_size = 12
    cell_size = 4
    grid_size = image_size // cell_size

    r = 5
    c = 8
    rc = np.array([[r, c]])

    theoretical_result = np.zeros((grid_size, grid_size, 3))

    pos_r = r // cell_size
    pos_c = c // cell_size

    theoretical_result[pos_r, pos_c] = (
        1,
        (r - pos_r * cell_size) / cell_size,
        (c - pos_c * cell_size) / cell_size,
    )
    output = get_prediction_matrix(rc, image_size=image_size, cell_size=cell_size)
    assert (theoretical_result == output).all()

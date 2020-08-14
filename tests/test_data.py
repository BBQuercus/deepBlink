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
from deepblink.data import normalize_images
from deepblink.data import get_coordinate_list
from deepblink.data import get_prediction_matrix


@pytest.mark.parametrize("value, base, expected", [(5, 2, 8), (6, 3, 9), (12, 7, 49)])
def test_next_power(value, base, expected):
    assert next_power(value, base) == expected


@pytest.mark.parametrize(
    "value, dividend, expected", [(5, 2, 6), (6, 3, 6), (8, 7, 14)]
)
def test_next_multiple(value, dividend, expected):
    assert next_multiple(value, dividend) == expected


@given(arrays(np.float, (5, 5), elements=floats(0, 100)))
def test_normalize_images(matrix):
    assert ((normalize_images(matrix) <= 1)).all()


def test_get_coordinate_list():
    grid_size = 4
    img_size = 512

    mat = np.zeros((grid_size, grid_size, 3))
    mat[0, 1, ...] = [1, 0.5, 0.5]

    theoretical_result = np.array(
        [img_size // grid_size * (1 + 0.5), img_size // grid_size * 0.5]
    )
    assert (get_coordinate_list(mat, size_image=img_size) == theoretical_result).all()


def test_get_prediction_matrix():
    img_size = 12
    cell_size = 4
    x = 5
    y = 8
    xy = np.array([[x, y]])
    grid_size = img_size // cell_size

    theoretical_result = np.zeros((grid_size, grid_size, 3))

    posx = x // cell_size
    posy = y // cell_size

    theoretical_result[posx, posy, ...] = [
        1,
        (x - posx * cell_size) / cell_size,
        (y - posy * cell_size) / cell_size,
    ]
    assert (theoretical_result == get_prediction_matrix(xy, img_size, cell_size)).all()

"""Unittests for the deepblink.augment module."""
# pylint: disable=missing-function-docstring

from hypothesis import given
from hypothesis.extra.numpy import arrays
import numpy as np
import pytest

from deepblink.augment import augment_batch_baseline
from deepblink.augment import flip
from deepblink.augment import gaussian_noise
from deepblink.augment import illuminate
from deepblink.augment import rotate
from deepblink.augment import translate


@given(arrays(np.float32, (3, 5, 5)))
def test_augment_batch_baseline(arr):
    imgs, masks = augment_batch_baseline(arr, arr)
    assert imgs.shape == masks.shape == arr.shape

    with pytest.warns(UserWarning):
        misshaped_arr = np.zeros((10, 5, 5))
        augment_batch_baseline(misshaped_arr, misshaped_arr)


@given(arrays(np.int8, (5, 5)))
def test_flip(matrix):
    img, mask = flip(matrix, matrix)
    assert np.sum(np.sum(img)) == np.sum(np.sum(matrix))
    assert mask.shape == matrix.shape


@given(arrays(np.int8, (5, 5)))
def test_illuminate(matrix):
    img, mask = illuminate(matrix, matrix)
    assert img.shape == matrix.shape
    assert mask.shape == matrix.shape


@given(arrays(np.int8, (5, 5)))
def test_gaussian_noise(matrix):
    img, mask = gaussian_noise(matrix, matrix)
    assert img.shape == matrix.shape
    assert mask.shape == matrix.shape


@given(arrays(np.int8, (5, 5)))
def test_rotate(matrix):
    img, mask = rotate(matrix, matrix)
    assert np.sum(np.sum(img)) == np.sum(np.sum(matrix))
    assert mask.shape == matrix.shape


@given(arrays(np.int8, (5, 5)))
def test_translate(matrix):
    img, mask = translate(matrix, matrix)
    assert np.sum(np.sum(img)) == np.sum(np.sum(matrix))
    assert mask.shape == matrix.shape

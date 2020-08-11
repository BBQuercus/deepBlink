"""Tests for model utility functions for augmentation."""

import numpy as np

from hypothesis.extra.numpy import arrays
from hypothesis import given

from deepblink.augment import flip
from deepblink.augment import illuminate
from deepblink.augment import gaussian_noise
from deepblink.augment import rotate
from deepblink.augment import translate


@given(arrays(np.int8, (5, 5)))
def test_flip(matrix):
    """Test function that flips an image."""
    img, _ = flip(matrix, matrix)
    assert np.sum(np.sum(img)) == np.sum(np.sum(matrix))


@given(arrays(np.int8, (5, 5)))
def test_illuminate(matrix):
    """Test function that adds illumation correction to image."""
    img, _ = illuminate(matrix, matrix)
    assert img.shape == matrix.shape


@given(arrays(np.int8, (5, 5)))
def test_gaussian_noise(matrix):
    """Test function that adds gaussian noise to image."""
    img, _ = gaussian_noise(matrix, matrix)
    assert img.shape == matrix.shape


@given(arrays(np.int8, (5, 5)))
def test_rotate(matrix):
    """Test function that rotates image."""
    img, _ = rotate(matrix, matrix)
    assert np.sum(np.sum(img)) == np.sum(np.sum(matrix))


@given(arrays(np.int8, (5, 5)))
def test_translate(matrix):
    """Test function that translates image."""
    img, _ = translate(matrix, matrix)
    assert np.sum(np.sum(img)) == np.sum(np.sum(matrix))

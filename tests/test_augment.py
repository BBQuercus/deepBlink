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
    img, mask = flip(matrix, matrix)
    assert np.sum(np.sum(img)) == np.sum(np.sum(matrix))
    assert mask.shape == matrix.shape


@given(arrays(np.int8, (5, 5)))
def test_illuminate(matrix):
    """Test function that adds illumation correction to image."""
    img, mask = illuminate(matrix, matrix)
    assert img.shape == matrix.shape
    assert mask.shape == matrix.shape


@given(arrays(np.int8, (5, 5)))
def test_gaussian_noise(matrix):
    """Test function that adds gaussian noise to image."""
    img, mask = gaussian_noise(matrix, matrix)
    assert img.shape == matrix.shape
    assert mask.shape == matrix.shape


@given(arrays(np.int8, (5, 5)))
def test_rotate(matrix):
    """Test function that rotates image."""
    img, mask = rotate(matrix, matrix)
    assert np.sum(np.sum(img)) == np.sum(np.sum(matrix))
    assert mask.shape == matrix.shape


@given(arrays(np.int8, (5, 5)))
def test_translate(matrix):
    """Test function that translates image."""
    img, mask = translate(matrix, matrix)
    assert np.sum(np.sum(img)) == np.sum(np.sum(matrix))
    assert mask.shape == matrix.shape

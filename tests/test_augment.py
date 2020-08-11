"""Tests for model utility functions for augmentation."""

import numpy as np

from hypothesis.extra.numpy import arrays
from hypothesis import given

from deepblink.augment import flip, illuminate, gaussian_noise, rotate, translate


@given(arrays(np.int8, (5, 5)))
def test_flip(matrix):
    """Test flip function."""
    img, _ = flip(matrix, matrix)
    assert np.sum(np.sum(img)) == np.sum(np.sum(matrix))


@given(arrays(np.int8, (5, 5)))
def test_illuminate(matrix):
    """Test illuminate function."""
    img, _ = illuminate(matrix, matrix)
    assert img.shape == matrix.shape


@given(arrays(np.int8, (5, 5)))
def test_gaussian_noise(matrix):
    """Test gaussian noise function."""
    img, _ = gaussian_noise(matrix, matrix)
    assert img.shape == matrix.shape


@given(arrays(np.int8, (5, 5)))
def test_rotate(matrix):
    """Test rotate function."""
    img, _ = rotate(matrix, matrix)
    assert np.sum(np.sum(img)) == np.sum(np.sum(matrix))


@given(arrays(np.int8, (5, 5)))
def test_translate(matrix):
    """Test translate function."""
    img, _ = translate(matrix, matrix)
    assert np.sum(np.sum(img)) == np.sum(np.sum(matrix))

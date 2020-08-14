"""Unittests for the deepblink.io module."""
# pylint: disable=missing-function-docstring

import os
import tempfile

from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from hypothesis.strategies import lists
import numpy as np
import pytest

from deepblink.io import extract_basename
from deepblink.io import load_npz
from deepblink.io import remove_zeros


@given(lists(arrays(np.float, (5, 5), elements=floats(0, 1)), min_size=3, max_size=10))
def test_remove_zeros(mylist):
    mylist[1] = 0
    assert len(remove_zeros(mylist)) == (len(mylist) - 1)


@pytest.mark.parametrize(
    "path, basename",
    [
        ("/dir/dir/file.ext", "file"),
        ("./../../file.ext.txt", "file.ext"),
        ("file", "file"),
    ],
)
def test_extract_basename(path, basename):
    assert extract_basename(path) == basename


def test_load_npz():
    with tempfile.TemporaryDirectory() as temp_dir:
        arr = np.zeros((3, 5, 5))
        fname = os.path.join(temp_dir, "file.npz")
        np.savez(
            fname,
            x_train=arr,
            y_train=arr,
            x_valid=arr,
            y_valid=arr,
            x_test=arr,
            y_test=arr,
        )

        data = load_npz(fname)
        assert len(data) == 6
        for d in data:
            assert d.shape == arr.shape

"""Unittests for the deepblink.io module."""
# pylint: disable=missing-function-docstring

from pathlib import Path
import os
import tempfile

import numpy as np
import pytest

from deepblink.io import basename
from deepblink.io import load_npz
from deepblink.io import grab_files


@pytest.mark.parametrize(
    "path, bname",
    [
        ("/dir/dir/file.ext", "file"),
        ("./../../file.ext.txt", "file.ext"),
        ("file", "file"),
    ],
)
def test_basename(path, bname):
    assert basename(path) == bname


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

        # Load all datasets
        data = load_npz(fname)
        assert len(data) == 6
        for d in data:
            assert d.shape == arr.shape

        # Load only test dataset
        data = load_npz(fname, test_only=True)
        assert len(data) == 2


def test_grab_files():
    """Test function that grabs files in a directory given the extensions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        fnames = ["test.txt", "text.csv", "test.h5", "csv.test"]
        for fname in fnames:
            Path(os.path.join(temp_dir, fname)).touch()

        ext = ["txt", "csv"]
        output = grab_files(temp_dir, ext)
        expected = [os.path.join(temp_dir, f) for f in ["test.txt", "text.csv"]]
        assert output == expected

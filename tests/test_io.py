"""Unittests for the deepblink.io module."""
# pylint: disable=missing-function-docstring

from pathlib import Path
import os
import tempfile

import numpy as np
import pytest
import skimage.io

from deepblink.io import basename
from deepblink.io import grab_files
from deepblink.io import load_image
from deepblink.io import load_npz
from deepblink.io import securename


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


@pytest.mark.parametrize(
    "fname, safename",
    [("hello.txt", "hello_txt"), ("./././hello", "______hello"), ("?#42|h", "__42_h")],
)
def test_securename(fname, safename):
    assert securename(fname) == safename


def test_load_image():
    with tempfile.TemporaryDirectory() as temp_dir:
        fname = os.path.join(temp_dir, "image.png")

        # Basic case
        arr = np.zeros((20, 20, 1))
        skimage.io.imsave(fname, arr)
        assert (load_image(fname, is_rgb=False) == arr.squeeze()).all()

        # RGB case
        arr = np.ones((20, 20, 3))
        skimage.io.imsave(fname, arr)
        assert load_image(fname, is_rgb=True).shape == (20, 20)

        # Multi-channel error
        with pytest.raises(ValueError):
            arr = np.zeros((20, 20, 20, 4))
            skimage.io.imsave(fname, arr)
            load_image(fname, is_rgb=False)


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

    with pytest.raises(OSError):
        grab_files("some_imaginary_dir/", ["txt"])

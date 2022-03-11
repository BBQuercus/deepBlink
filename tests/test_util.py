"""Unittests for the deepblink.util module."""
# pylint: disable=missing-function-docstring

import os

from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.strategies import lists
from hypothesis.extra.numpy import arrays
import numpy as np
import pandas as pd
import pytest

from deepblink.util import delete_non_unique_columns
from deepblink.util import get_from_module
from deepblink.util import predict_pixel_size
from deepblink.util import predict_shape
from deepblink.util import relative_shuffle
from deepblink.util import remove_falses
from deepblink.util import train_valid_split


def test_get_from_module():
    assert get_from_module("deepblink.util", "predict_shape") == predict_shape
    assert get_from_module("deepblink.util", "train_valid_split") == train_valid_split


@given(lists(floats(-100, 100), min_size=1, max_size=100))
def test_relative_shuffle_lists(mylist):
    x, y = relative_shuffle(mylist, mylist)
    assert len(x) == len(mylist)
    assert len(y) == len(mylist)
    assert sorted(x) == sorted(mylist)


@given(arrays(np.int8, (20, 3)))
def test_relative_shuffle_arrays(myarr):
    x, y = relative_shuffle(myarr, myarr)
    assert len(x) == len(myarr)
    assert len(y) == len(myarr)


@given(lists(floats(-100, 100), min_size=3, max_size=100))
def test_train_valid_split(mylist):
    split = 0.5
    xtrain, xvalid, ytrain, yvalid = train_valid_split(mylist, mylist, split)

    split_len = round(len(mylist) * split)
    assert (len(xvalid)) == split_len
    assert (len(xtrain)) == (len(mylist) - (split_len))

    assert (len(yvalid)) == (split_len)
    assert (len(ytrain)) == (len(mylist) - (split_len))


def test_delete_non_unique_columns():
    columns = list("ABCD")
    df = pd.DataFrame(np.random.randint(0, 100, size=(15, 4)), columns=columns)

    # Keep unique columns
    assert all(delete_non_unique_columns(df).columns == columns)

    # Delete non unique columns
    df["E"] = np.random.randint(100)
    df["F"] = "test"
    assert all(delete_non_unique_columns(df).columns == columns)


@pytest.mark.parametrize(
    "tupl, expected",
    [
        ((0, 1, 2), (1, 2)),
        ((0, 0), ()),
        ((False, True, None, 1), (True, 1)),
        ((1, 2), (1, 2)),
    ],
)
def test_remove_falses(tupl, expected):
    assert remove_falses(tupl) == expected


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((100, 100, 2), ("x,y,c")),
        ((100, 100, 3), ("x,y,3")),
        ((10, 100, 100), ("z,x,y")),
        ((10, 100, 100, 2), ("z,x,y,c")),
        ((20, 10, 100, 100), ("t,z,x,y")),
        ((20, 10, 100, 100, 2), ("t,z,x,y,c")),
    ],
)
def test_predict_shape_1(shape, expected):
    assert predict_shape(shape) == expected


@pytest.mark.parametrize("shape", [(100,), (0, 0), list(range(7))])
def test_predict_shape_2(shape):
    with pytest.raises(ValueError):
        predict_shape(shape)


def test_predict_pixel_size():
    images = [
        os.path.join(os.path.dirname(__file__), "./data/image_pixel.tif"),
        os.path.join(os.path.dirname(__file__), "./data/image_micron.tif"),
    ]
    assert predict_pixel_size(images[0]) == (1.0, 1.0)
    assert predict_pixel_size(images[1]) == pytest.approx((0.3878, 0.4824))

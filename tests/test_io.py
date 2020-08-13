"""Test dataset preparation functions."""

import numpy as np

from hypothesis.extra.numpy import arrays
from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.strategies import lists

from deepblink.io import remove_zeros
from deepblink.io import extract_basename


@given(lists(arrays(np.float, (5, 5), elements=floats(0, 1)), min_size=3, max_size=10))
def test_remove_zeros(mylist):
    """Test remove zeros from list of numpy arrays"""
    length = len(mylist)
    mylist[1] = 0
    assert len(remove_zeros(mylist)) == (length - 1)


def test_extract_basename():
    """Test function that extracts basename from absolute path."""
    abs_path = "/test/test/basename"
    assert extract_basename(abs_path) == "basename"

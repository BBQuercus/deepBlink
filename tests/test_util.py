"""Test utils functions."""

from hypothesis.strategies import lists
from hypothesis.strategies import floats
from hypothesis import given

from deepblink.util import relative_shuffle
from deepblink.util import train_valid_split


@given(lists(floats(-100, 100), min_size=1, max_size=100))
def test_relative_shuffle(mylist):
    """Test function that shuffles lists."""
    x, y = relative_shuffle(mylist, mylist)
    assert len(x) == len(mylist)
    assert len(y) == len(mylist)


@given(lists(floats(-100, 100), min_size=3, max_size=100))
def test_train_valid_split(mylist):
    """Test function that split dataset into train and validation."""
    split = 0.5
    xtrain, xvalid, ytrain, yvalid = train_valid_split(mylist, mylist, split)

    split_len = round(len(mylist) * split)
    assert (len(xvalid)) == split_len
    assert (len(xtrain)) == (len(mylist) - (split_len))

    assert (len(yvalid)) == (split_len)
    assert (len(ytrain)) == (len(mylist) - (split_len))

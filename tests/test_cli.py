"""Unittests for the deepblink.cli module."""
# pylint: disable=missing-function-docstring,redefined-outer-name

import os

import numpy as np
import pytest
import tensorflow as tf

from deepblink.cli import _grab_files
from deepblink.cli import _predict
from deepblink.losses import combined_f1_rsme
from deepblink.losses import f1_score
from deepblink.losses import rmse


@pytest.fixture
def data_dir():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


@pytest.fixture
def db_model(data_dir):
    fname = os.path.join(data_dir, "model512x512.h5")
    return tf.keras.models.load_model(
        fname,
        custom_objects={
            "f1_score": f1_score,
            "rmse": rmse,
            "combined_f1_rsme": combined_f1_rsme,
        },
    )


def test_grab_files(data_dir):
    """Test function that grabs files in a directory given the extensions."""
    ext = ["txt", "csv"]
    assert isinstance(_grab_files(data_dir, ext), list)


def test_predict(db_model):
    """Test the function that given model and image, returns the prediction on image."""
    img = np.random.rand(512, 512)
    pred = _predict(img, db_model)
    assert isinstance(pred, np.ndarray)

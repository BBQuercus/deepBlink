"""Test command line interface functions."""

import os

import numpy as np
import tensorflow as tf

from deepblink.cli import _grab_files
from deepblink.cli import _predict
from deepblink.cli import predict_baseline
from deepblink.losses import f1_l2_combined_loss
from deepblink.losses import f1_score
from deepblink.losses import l2_norm

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def test_grab_files():
    """Test function that grabs files in a directory given the extensions."""
    ext = ['txt', 'csv']
    path = os.path.join(CURRENT_DIR, 'data')

    assert isinstance(_grab_files(path, ext), list)


def test_predict():
    """Test the function that given model and image, returns the predicted coordinates."""
    img = np.random.rand(512, 512)
    model = os.path.join(CURRENT_DIR, 'data', 'model512x512.h5')
    model = tf.keras.models.load_model(
        model,
        custom_objects={
            "f1_score": f1_score,
            "l2_norm": l2_norm,
            "f1_l2_combined_loss": f1_l2_combined_loss,
        },
    )
    xcoord, ycoord = _predict(img, model)
    assert isinstance(xcoord, np.ndarray)
    assert isinstance(ycoord, np.ndarray)


def test_predict_baseline():
    """Test the function that given model and image, returns the prediction on image."""
    img = np.random.rand(512, 512)
    model = os.path.join(CURRENT_DIR, 'data', 'model512x512.h5')
    model = tf.keras.models.load_model(
        model,
        custom_objects={
            "f1_score": f1_score,
            "l2_norm": l2_norm,
            "f1_l2_combined_loss": f1_l2_combined_loss,
        },
    )
    pred = predict_baseline(img, model)
    assert isinstance(pred, np.ndarray)

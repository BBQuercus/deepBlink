"""Unittests for the deepblink.cli module."""
# pylint: disable=missing-function-docstring,redefined-outer-name

from pathlib import Path
import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from deepblink.cli import _grab_files
from deepblink.cli import _predict
from deepblink.losses import combined_bce_rmse
from deepblink.losses import combined_f1_rmse
from deepblink.losses import f1_score
from deepblink.losses import rmse


def test_grab_files():
    """Test function that grabs files in a directory given the extensions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        fnames = ["test.txt", "text.csv", "test.h5", "csv.test"]
        for fname in fnames:
            Path(os.path.join(temp_dir, fname)).touch()

        ext = ["txt", "csv"]
        output = _grab_files(temp_dir, ext)
        expected = [os.path.join(temp_dir, f) for f in ["test.txt", "text.csv"]]
        assert output == expected


def test_predict():
    """Test the function that given model and image, returns the prediction on image."""
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input((None, None, 1)),
            tf.keras.layers.Conv2D(3, 3, strides=2),
            tf.keras.layers.Activation("sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=combined_bce_rmse,
        metrics=[f1_score, rmse, combined_f1_rmse],
    )

    for size in [249, 512, 876]:
        img = np.random.rand(size, size)
        pred = _predict(img, model)
        assert isinstance(pred, np.ndarray)

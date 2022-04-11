"""Unittests for the deepblink.cli module."""
# pylint: disable=missing-function-docstring
from unittest import mock
import argparse
import logging
import os

import numpy as np
import pandas as pd
import pytest

from deepblink.cli._config import HandleConfig
from deepblink.cli._create import HandleCreate
from deepblink.cli._predict import HandlePredict
from deepblink.cli._main import arg_parser


LOGGER = logging.Logger("None")


# TODO cover more in-depth functionality like commands etc.
def test_parse_args():
    parser = arg_parser()

    assert isinstance(parser, argparse.ArgumentParser)

    with mock.patch("sys.argv", [""]):
        assert isinstance(parser.parse_args(), argparse.Namespace)


def test_config():
    temp_file = "test"
    full_file = os.path.abspath(temp_file + ".yaml")
    try:
        handler = HandleConfig(arg_name=temp_file, arg_raw=False, logger=LOGGER)
        handler()
        assert os.path.exists(full_file)
    finally:
        os.remove(full_file)


# TODO image_label_lists, crop_image, get_file_lists, splits, save_npz
def test_create():
    handler = HandleCreate(
        arg_input="None",
        arg_labels="None",
        arg_name="None",
        arg_size=0,
        arg_pixel_size=1.0,
        arg_testsplit=0,
        arg_validsplit=0,
        arg_minspots=0,
        logger=LOGGER,
    )

    labels = [
        os.path.join(os.path.dirname(__file__), "./data/input_manual.csv"),
        os.path.join(os.path.dirname(__file__), "./data/input_trackmate_post7.csv"),
        os.path.join(os.path.dirname(__file__), "./data/input_trackmate_pre7.csv"),
    ]
    max_size = 100
    image = np.zeros((max_size, max_size))

    for label in labels:
        df = pd.read_csv(label, index_col=0)
        df = handler.convert_labels(image, df, (1.0, 1.0))
        assert df.columns.to_list() == ["r", "c"]
        assert df.dtypes.to_list() == [np.float64, np.float64]
        assert df["r"].min() >= 0
        assert df["c"].min() >= 0
        assert df["r"].max() <= max_size
        assert df["c"].max() <= max_size


# TODO test batch prediction
@pytest.fixture(scope="session")
def predict_handler():
    handler = HandlePredict(
        arg_input=os.path.join(os.path.dirname(__file__), "./data/image_pixel.tif"),
        arg_model=os.path.join(os.path.dirname(__file__), "./data/model.h5"),
        arg_output=None,
        arg_radius=None,
        arg_shape=None,
        arg_probability=None,
        arg_pixel_size=None,
        logger=LOGGER,
    )
    return handler


@pytest.fixture(scope="session")
def filename_predict_output():
    return os.path.join(os.path.dirname(__file__), "./data/image_pixel.csv")


def test_predict_different_name(predict_handler):
    filename_special = os.path.join(os.getcwd(), "image_pixel.csv")
    predict_handler.raw_output = os.getcwd()
    predict_handler()
    assert os.path.exists(filename_special)
    os.remove(filename_special)
    predict_handler.raw_output = None


# Test radius and no output -> same name as input
def test_predict_radius_and_output(predict_handler, filename_predict_output):
    predict_handler.radius = 1
    predict_handler()
    assert os.path.exists(filename_predict_output)
    with open(filename_predict_output, "r") as f:
        assert f.readline().strip() == "y [px],x [px],i"
    os.remove(filename_predict_output)
    predict_handler.radius = None


def test_predict_probability_and_pixel_size(predict_handler, filename_predict_output):
    predict_handler.pixel_size = 2.3
    predict_handler.probability = 0.9
    predict_handler()
    with open(filename_predict_output, "r") as f:
        assert f.readline().strip() == "y [\u00B5m],x [\u00B5m],p"
    os.remove(filename_predict_output)
    predict_handler.pixel_size = None
    predict_handler.probability = None


def test_predict_all_options(predict_handler, filename_predict_output):
    predict_handler.pixel_size = 2.3
    predict_handler.probability = 0.9
    predict_handler.radius = 3
    predict_handler.raw_shape = "(x,y)"
    predict_handler()
    with open(filename_predict_output, "r") as f:
        assert f.readline().strip() == "y [\u00B5m],x [\u00B5m],p,i"
    os.remove(filename_predict_output)
    predict_handler.pixel_size = None
    predict_handler.probability = None
    predict_handler.radius = None
    predict_handler.raw_shape = None

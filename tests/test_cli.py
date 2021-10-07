"""Unittests for the deepblink.cli module."""
# pylint: disable=missing-function-docstring
from unittest import mock
import argparse
import logging
import os

import numpy as np
import pandas as pd

from deepblink.cli._config import HandleConfig
from deepblink.cli._create import HandleCreate
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
        df = handler.convert_labels(image, df)
        assert df.columns.to_list() == ["r", "c"]
        assert df.dtypes.to_list() == [np.float64, np.float64]
        assert df["r"].min() >= 0
        assert df["c"].min() >= 0
        assert df["r"].max() <= max_size
        assert df["c"].max() <= max_size

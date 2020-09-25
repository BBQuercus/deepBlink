"""Unittests for the deepblink.cli module."""
# pylint: disable=missing-function-docstring
from unittest import mock
import argparse

from deepblink.cli._main import arg_parser


# TODO cover more in-depth functionality like commands etc.
def test_parse_args():
    parser = arg_parser()

    assert isinstance(parser, argparse.ArgumentParser)

    with mock.patch("sys.argv", [""]):
        assert isinstance(parser.parse_args(), argparse.Namespace)

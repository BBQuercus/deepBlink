"""Unittests for the deepblink.metrics module."""
# pylint: disable=missing-function-docstring

from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
import pytest
import scipy.spatial

from deepblink.metrics import _f1_at_cutoff
from deepblink.metrics import error_on_coordinates
from deepblink.metrics import euclidean_dist
from deepblink.metrics import f1_integral
from deepblink.metrics import f1_score
from deepblink.metrics import linear_sum_assignment
from deepblink.metrics import offset_euclidean
from deepblink.metrics import precision_score
from deepblink.metrics import recall_score


@pytest.mark.parametrize(
    "x1, y1, x2, y2, expected",
    [(2, 2, 2, 2, 0), (2, 2, 2, 3, 1), (3, 3, 4, 4, np.sqrt(2))],
)
def test_euclidean_dist(x1, y1, x2, y2, expected):
    assert euclidean_dist(x1, y1, x2, y2) == expected


@pytest.mark.parametrize(
    "nfalsepositive, expected_precision", [(2, 0.98), (3, 0.97), (0, 1), (7, 0.93)]
)
def test_precision_score(nfalsepositive, expected_precision):
    true = np.ones((10, 10, 3))
    pred = np.ones((10, 10, 3))

    index1 = np.random.choice(true.shape[0], nfalsepositive, replace=False)
    index2 = np.random.choice(true.shape[0], nfalsepositive, replace=False)
    true[index1, index2, 0] = 0

    assert precision_score(pred, true) == expected_precision


@pytest.mark.parametrize(
    "nfalsenegative, expected_recall", [(2, 0.98), (3, 0.97), (0, 1), (7, 0.93)]
)
def test_recall_score(nfalsenegative, expected_recall):
    true = np.ones((10, 10, 3))
    pred = np.ones((10, 10, 3))

    index1 = np.random.choice(true.shape[0], nfalsenegative, replace=False)
    index2 = np.random.choice(true.shape[0], nfalsenegative, replace=False)
    pred[index1, index2, 0] = 0

    assert recall_score(pred, true) == expected_recall


@pytest.mark.parametrize(
    "nfalsenegative, expected_recall", [(2, 0.98), (3, 0.97), (0, 1), (7, 0.93)]
)
def test_f1_score(nfalsenegative, expected_recall):
    true = np.ones((10, 10, 3))
    pred = np.ones((10, 10, 3))

    index1 = np.random.choice(true.shape[0], nfalsenegative, replace=False)
    index2 = np.random.choice(true.shape[0], nfalsenegative, replace=False)
    pred[index1, index2, 0] = 0
    output = f1_score(pred, true)
    expected = (2 * expected_recall * 1) / (expected_recall + 1)
    assert output == pytest.approx(expected)


@pytest.mark.parametrize("ndifferent, expected_error", [(2, 0.016), (5, 0.04), (0, 0)])
def test_error_on_coordinates(ndifferent, expected_error):
    true = np.ones((10, 10, 3))
    true[..., 1] = 0.5
    pred = np.ones((10, 10, 3))
    pred[..., 1] = 0.5

    index1 = np.random.choice(true.shape[0], ndifferent, replace=False)
    index2 = np.random.choice(true.shape[0], ndifferent, replace=False)
    pred[index1, index2, 1] = 0.3

    assert error_on_coordinates(pred, true, 4) == expected_error


def test_linear_sum_assignment():
    distance_matrix = np.ones((10, 10))
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix[0, 0] = 1
    expected_result = ([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert linear_sum_assignment(distance_matrix, cutoff=0.001) == expected_result


@pytest.mark.filterwarnings("ignore::FutureWarning")
@given(arrays(np.int8, (10, 2)))
def test_f1_cutoff_score(true):
    pred = true + np.ones((10, 2))
    assert f1_cutoff_score(pred, true, 2) == 1
    assert f1_cutoff_score(pred, true, 0) == 0

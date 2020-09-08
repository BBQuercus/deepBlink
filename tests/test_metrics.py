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


@given(n=st.integers(min_value=0, max_value=20))
def test_linear_sum_assignment_diagonal(n):
    # Basic diagonal matrix with lowest scores along diagonal
    matrix = 1 - np.diag(np.ones(n))
    output = linear_sum_assignment(matrix, cutoff=0)
    expected = (list(range(n)), list(range(n)))
    assert output == expected


def test_linear_sum_assignment_non_diagonal():
    # Offset diagonal matrix
    matrix = 1 - np.diag(np.ones(3))
    matrix[:, [0, 1]] = matrix[:, [1, 0]]
    output = linear_sum_assignment(matrix, cutoff=0)
    expected = (list(range(3)), [1, 0, 2])
    assert output == expected


def test_f1_at_cutoff():
    true = np.zeros((10, 2))
    pred = true + 1

    # Without offset
    matrix = scipy.spatial.distance.cdist(pred, true, metric="euclidean")

    for cutoff, expected in zip([0, 1, 2], [0, 0, 1]):
        output = _f1_at_cutoff(matrix, pred, true, cutoff=cutoff, return_raw=False)
        assert output == pytest.approx(expected)

    # # With empty offset
    output = _f1_at_cutoff(matrix, pred, true, cutoff=0, return_raw=True)
    assert len(output) == 3
    assert output[1] == []

    # # With populated offset
    output = _f1_at_cutoff(matrix, pred, true, cutoff=2, return_raw=True)
    assert len(output) == 3
    assert sorted(output[1]) == list(range(10))
    assert sorted(output[2]) == list(range(10))


@given(arrays(np.uint8, (10, 2)))
def test_f1_integral_v0(true):
    # Equal inputs
    output = f1_integral(true, true, max_distance=5, n_cutoffs=20, return_raw=False)
    assert output == pytest.approx(1)


def test_f1_integral_v1():
    # Unequal inputs
    true = np.ones((10, 2))
    pred = np.zeros((10, 2))
    output = f1_integral(pred, true, max_distance=2, n_cutoffs=11, return_raw=False)
    assert output == pytest.approx(0.25)

    # Raw output
    output = f1_integral(true, true, max_distance=5, n_cutoffs=20, return_raw=True)
    assert len(output) == 3
    assert len(output[1]) == 20  # 20 cutoffs
    assert len(output[1][0]) == 10  # 10 coord inputs -> offsets
    assert (output[2] == np.linspace(start=0, stop=5, num=20)).all()


def test_offset_euclidean():
    offset = [[0, 0], [1, 1], [-1, 1], [1, 0]]
    expected = [0, np.sqrt(2), np.sqrt(2), 1]
    assert offset_euclidean(offset) == pytest.approx(expected)

"""Metrics functions."""

import numpy as np
import pandas as pd


def euclidean_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return the euclidean distance between two points."""
    return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))


def precision(pred: np.ndarray, true: np.ndarray) -> float:
    """Returns the precision defined as (True positive)/(True positive + False positive).

    Precision will be measured within cell of size "cell_size". If cell_size = 1, precision
    will be measured at resolution of pixel.

    Note – direction dependent, arguments cant be switched!!

    Args:
        pred: np.ndarray of shape (n, n, 3): p, x, y format for each cell.
        true: np.ndarray of shape (n, n, 3): p, x, y format for each cell.

    Returns:
        Precision score.
    """
    selection = pred[..., 0] == 1
    p = np.mean(true[selection, 0])
    return p


def recall(pred: np.ndarray, true: np.ndarray) -> float:
    """Returns the recall defined as (True positive)/(True positive + False negative).

    Recall will be measured within cell of size "cell_size". If cell_size = 1, recall
    will be measured at resolution of pixel.

    Note – direction dependent, arguments cant be switched!!

    Args:
        pred: np.ndarray of shape (n, n, 3): p, x, y format for each cell.
        true: np.ndarray of shape (n, n, 3): p, x, y format for each cell.

    Returns:
        Recall score.
    """
    selection = true[..., 0] == 1
    r = np.mean(pred[selection, 0])
    return r


def f1_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Returns F1 score defined as: 2 * precision*recall / precision+recall.

    F1 score will be measured within cell of size "cell_size". If cell_size = 1, F1 score
    will be measured at resolution of pixel.

    Note – direction dependent, arguments cant be switched!!

    Args:
        pred: np.ndarray of shape (n, n, 3): p, x, y format for each cell.
        true: np.ndarray of shape (n, n, 3): p, x, y format for each cell.

    Returns:
        F1 score.
    """
    r = recall(pred, true)
    p = precision(pred, true)

    if r == 0 and p == 0:
        return 0

    f1_score_ = 2 * p * r / (p + r)
    return f1_score_


def error_on_coordinates(pred: np.ndarray, true: np.ndarray, cell_size: int) -> float:
    """Calculate the average error on spot coordinates.

    Args:
        pred: np.ndarray of shape (n, n, 3): p, x, y format for each cell.
        true: np.ndarray of shape (n, n, 3): p, x, y format for each cell.
        cell_size: Size of cell used to calculate F1 score, precision and recall.

    Returns:
        Error on coordinate.
    """
    spot = (true[..., 0] == 1) & (pred[..., 0] == 1)
    d = 0.0
    counter = 0
    assert pred.shape == true.shape

    for i in range(len(pred)):
        for j in range(len(pred)):
            if spot[i, j]:
                x1 = true[i, j, 1] * cell_size
                x2 = pred[i, j, 1] * cell_size
                y1 = true[i, j, 2] * cell_size
                y2 = pred[i, j, 2] * cell_size
                d += euclidean_dist(x1=x1, y1=y1, x2=x2, y2=y2)
                counter += 1

    if counter:
        d = d / counter
    else:
        d = None  # type: ignore

    return d


def weighted_f1_coordinates(
    pred: np.ndarray, true: np.ndarray, cell_size: int, weight: float = 1
) -> float:
    """Returns weighted single score defined as: weight*(1-F1) + (error on coordinate).

    F1 score will be measured within cell of size "cell_size". If cell_size = 1, F1 score
    will be measured at resolution of pixel.

    Note – direction dependent, arguments cant be switched!!

    Args:
        pred: np.ndarray of shape (n, n, 3): p, x, y format for each cell.
        true: np.ndarray of shape (n, n, 3): p, x, y format for each cell
        cell_size: Size of cells in the grid used to calculate F1 score, relative coordinates.
        weight: Weight of 1-F1 score in the average default = 1.

    Returns:
        Weighted score.
    """
    f1_score_ = f1_score(pred, true)
    f1_score_ = (1.0 - f1_score_) * weight

    error_coordinates = error_on_coordinates(pred, true, cell_size)
    if error_coordinates is not None:
        score = (f1_score_ + error_coordinates) / 2
        return score

    return None


def compute_score(
    true: np.ndarray, pred: np.ndarray, cell_size: int, weight: float
) -> pd.DataFrame:
    """Compute F1 score, error on coordinate and a weighted average of the two.

    Note – direction dependent, arguments cant be switched!!

    Args:
        pred: list of np.ndarray of shape (n, n, 3): p, x, y format for each cell.
        true: list of np.ndarray of shape (n, n, 3): p, x, y format for each cell.
        cell_size: Size of cells in the grid used to calculate F1 score, relative coordinates.
        weight: Weight to on f1 score.

    Returns:
        DataFrame with all three columns corresponding to f1 score, coordinate error, and weighted average.
    """
    f1_score_ = pd.Series()
    error_on_coordinates_ = pd.Series()
    weighted_f1_coordinates_ = pd.Series()

    for p, t in zip(true, pred):
        f1_score_.append(f1_score(p, t))
        error_on_coordinates_.append(error_on_coordinates(p, t, cell_size))
        weighted_f1_coordinates_.append(
            weighted_f1_coordinates(p, t, cell_size, weight)
        )

    df = pd.DataFrame([f1_score_, error_on_coordinates_, weighted_f1_coordinates_]).T
    df.columns = ["f1_score", "err_coordinate", "weighted_average"]
    return df

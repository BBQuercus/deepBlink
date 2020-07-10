"""Functions to calculate training loss on single image."""

from typing import Optional

import numpy as np
import pandas as pd


def euclidean_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return the euclidean distance between two the points (x1, y1) and (x2, y2)."""
    return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))


def precision_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Precision score metric.

    Defined as ``tp / (tp + fp)`` where tp is the number of true positives and fp the number of false positives.
    Can be interpreted as the accuracy to not mislabel samples or how many selected items are relevant.
    The best value is 1 and the worst value is 0.

    NOTE – direction dependent, arguments cant be switched!!

    Args:
        pred: np.ndarray of shape (n, n, 3): p, x, y format for each cell.
        true: np.ndarray of shape (n, n, 3): p, x, y format for each cell.
    """
    selection = pred[..., 0] == 1
    precision = np.mean(true[selection, 0])
    return precision


def recall_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Recall score metric.

    Defined as ``tp / (tp + fn)`` where tp is the number of true positives and fn the number of false negatives.
    Can be interpreted as the accuracy of finding positive samples or how many relevant samples were selected.
    The best value is 1 and the worst value is 0.

    NOTE – direction dependent, arguments cant be switched!!

    Args:
        pred: np.ndarray of shape (n, n, 3): p, x, y format for each cell.
        true: np.ndarray of shape (n, n, 3): p, x, y format for each cell.
    """
    selection = true[..., 0] == 1
    recall = np.mean(pred[selection, 0])
    return recall


def f1_score(pred: np.ndarray, true: np.ndarray) -> Optional[float]:
    r"""F1 score metric.

    .. math::
        F1 = \frac{2 * precision * recall} / {precision + recall}.

    The equally weighted average of precision and recall.
    The best value is 1 and the worst value is 0.

    NOTE – direction dependent, arguments cant be switched!!

    Args:
        pred: np.ndarray of shape (n, n, 3): p, x, y format for each cell.
        true: np.ndarray of shape (n, n, 3): p, x, y format for each cell.
    """
    recall = recall_score(pred, true)
    precision = precision_score(pred, true)

    if recall == 0 and precision == 0:
        return None

    f1_value = (2 * precision * recall) / (precision + recall)
    return f1_value


# TODO find better name
def error_on_coordinates(
    pred: np.ndarray, true: np.ndarray, cell_size: int
) -> Optional[float]:
    """The mean error on spot coordinates.

    F1 score will be measured within cell of size "cell_size".
    If cell_size = 1, F1 score will be measured at resolution of pixel.

    Args:
        pred: np.ndarray of shape (n, n, 3): p, x, y format for each cell.
        true: np.ndarray of shape (n, n, 3): p, x, y format for each cell.
        cell_size: Size of cell used to calculate F1 score, precision and recall.

    Returns:
        If no spots are found None, else the error on coordinate.
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
        return d / counter

    return None


# TODO find better name
def weighted_f1_coordinates(
    pred: np.ndarray, true: np.ndarray, cell_size: int, weight: float = 1
) -> Optional[float]:
    """A single weighted score defined as ``weight*(F1 loss) + (error on coordinate)``.

    F1 score will be measured within cell of size "cell_size".
    If cell_size = 1, F1 score will be measured at resolution of pixel.
    NOTE – direction dependent, arguments cant be switched!!

    Args:
        pred: np.ndarray of shape (n, n, 3): p, x, y format for each cell.
        true: np.ndarray of shape (n, n, 3): p, x, y format for each cell
        cell_size: Size of cells in the grid used to calculate F1 score, relative coordinates.
        weight: Weight of 1-F1 score in the average default = 1.
    """
    f1_value = f1_score(pred, true)
    error_coordinates = error_on_coordinates(pred, true, cell_size)

    if f1_value is not None and error_coordinates is not None:
        f1_value = (1.0 - f1_value) * weight
        score = (f1_value + error_coordinates) / 2
        return score

    return None


# TODO find better name
def compute_score(
    true: np.ndarray, pred: np.ndarray, cell_size: int, weight: float
) -> pd.DataFrame:
    """Compute F1 score, error on coordinate and a weighted average of the two.

    F1 score will be measured within cell of size "cell_size".
    If cell_size = 1, F1 score will be measured at resolution of pixel.
    NOTE – direction dependent, arguments cant be switched!!

    Args:
        pred: list of np.ndarray of shape (n, n, 3): p, x, y format for each cell.
        true: list of np.ndarray of shape (n, n, 3): p, x, y format for each cell.
        cell_size: Size of cells in the grid used to calculate F1 score, relative coordinates.
        weight: Weight to on f1 score.

    Returns:
        DataFrame with all three columns corresponding to f1 score, coordinate error, and weighted average.
    """
    f1_value = pd.Series()
    error_on_coordinates_ = pd.Series()
    weighted_f1_coordinates_ = pd.Series()

    for t, p in zip(true, pred):
        f1_value.append(f1_score(p, t))
        error_on_coordinates_.append(error_on_coordinates(p, t, cell_size))
        weighted_f1_coordinates_.append(
            weighted_f1_coordinates(p, t, cell_size, weight)
        )

    df = pd.DataFrame([f1_value, error_on_coordinates_, weighted_f1_coordinates_]).T
    df.columns = ["f1_score", "err_coordinate", "weighted_average"]
    return df

"""Functions to calculate training loss on single image."""

from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import scipy.optimize

EPS = 1e-12


def euclidean_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return the euclidean distance between two the points (x1, y1) and (x2, y2)."""
    return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))


def offset_euclidean(offset: List[tuple]) -> np.ndarray:
    """Calculates the euclidean distance based on row_column_offsets per coordinate."""
    return np.sqrt(np.sum(np.square(np.array(offset)), axis=-1))


def precision_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Precision score metric.

    Defined as ``tp / (tp + fp)`` where tp is the number of true positives and fp the number of false positives.
    Can be interpreted as the accuracy to not mislabel samples or how many selected items are relevant.
    The best value is 1 and the worst value is 0.

    NOTE – direction dependent, arguments cant be switched!!

    Args:
        pred: np.ndarray of shape (n, n, 3): p, r, c format for each cell.
        true: np.ndarray of shape (n, n, 3): p, r, c format for each cell.
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
        pred: np.ndarray of shape (n, n, 3): p, r, c format for each cell.
        true: np.ndarray of shape (n, n, 3): p, r, c format for each cell.
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
        pred: np.ndarray of shape (n, n, 3): p, r, c format for each cell.
        true: np.ndarray of shape (n, n, 3): p, r, c format for each cell.
    """
    recall = recall_score(pred, true)
    precision = precision_score(pred, true)

    if recall == 0 and precision == 0:
        return None

    f1_value = (2 * precision * recall) / (precision + recall + EPS)
    return f1_value


# TODO remove test on depreciation
def error_on_coordinates(
    pred: np.ndarray, true: np.ndarray, cell_size: int
) -> Optional[float]:
    """The mean error on spot coordinates.

    F1 score will be measured within cell of size "cell_size".
    If cell_size = 1, F1 score will be measured at resolution of pixel.

    Args:
        pred: np.ndarray of shape (n, n, 3): p, r, c format for each cell.
        true: np.ndarray of shape (n, n, 3): p, r, c format for each cell.
        cell_size: Size of cell used to calculate F1 score, precision and recall.

    Returns:
        If no spots are found None, else the error on coordinate.
    """
    warnings.warn(
        "deepblink.metrics.error_on_coordinates will be depreciated in the next release.",
        DeprecationWarning,
    )

    spot = (true[..., 0] == 1) & (pred[..., 0] == 1)
    d = 0.0
    counter = 0
    assert pred.shape == true.shape

    row, col = np.asarray(spot).nonzero()
    for i, j in zip(row, col):
        x1 = true[i, j, 1] * cell_size
        x2 = pred[i, j, 1] * cell_size
        y1 = true[i, j, 2] * cell_size
        y2 = pred[i, j, 2] * cell_size
        d += euclidean_dist(x1=x1, y1=y1, x2=x2, y2=y2)
        counter += 1

    if counter:
        return d / counter

    return None


def linear_sum_assignment(
    matrix: np.ndarray, cutoff: float = None
) -> Tuple[list, list]:
    """Solve the linear sum assignment problem with a cutoff.

    A problem instance is described by matrix matrix where each matrix[i, j]
    is the cost of matching i (worker) with j (job). The goal is to find the
    most optimal assignment of j to i if the given cost is below the cutoff.

    Args:
        matrix: Matrix containing cost/distance to assign cols to rows.
        cutoff: Maximum cost/distance value assignments can have.

    Returns:
        (rows, columns) corresponding to the matching assignment.
    """
    # Prevent scipy to optimize on values above the cutoff
    if cutoff is not None:
        matrix = np.where(matrix >= cutoff, matrix.max(), matrix)

    row, col = scipy.optimize.linear_sum_assignment(matrix)

    # Allow for no assignment based on cutoff
    if cutoff is not None:
        nrow = []
        ncol = []
        for r, c in zip(row, col):
            if matrix[r, c] <= cutoff:
                nrow.append(r)
                ncol.append(c)
        return nrow, ncol

    return list(row), list(col)


def f1_cutoff_score(pred: np.ndarray, true: np.ndarray, cutoff: float = None) -> float:
    """Alternative way of F1 score computation.

    Computes a distance matrix between every coordinate.
    Based on the best assignment below a cutoff (coordinate closeness),
    corresponding precision and recall is calculated.

    Args:
        pred: Array of shape (n, 2) for predicted coordinates.
        true: Array of shape (n, 2) for ground truth coordinates.
        cutoff: Distance cutoff to allow coordinate assignment.

    Returns:
        F1 score metric.
    """
    warnings.warn(
        "F1 cuttoff score is a test function and might be depreciated in the next release.",
        FutureWarning,
    )

    matrix = scipy.spatial.distance.cdist(pred, true, metric="euclidean")

    pred_true = linear_sum_assignment(matrix, cutoff)[0]
    true_pred = linear_sum_assignment(matrix.T, cutoff)[0]

    if not pred_true:
        return 0.0

    tp = len(true_pred)
    fn = len(true) - len(true_pred)
    fp = len(pred) - len(pred_true)

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_value = (2 * precision * recall) / (precision + recall)

    return f1_value

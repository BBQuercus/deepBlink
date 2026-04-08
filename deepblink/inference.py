"""Model prediction / inference functions."""

import math

import numpy as np
import skimage.morphology
import keras

from .data import get_coordinate_list
from .data import next_power
from .data import normalize_image


def predict(
    image: np.ndarray,
    model: keras.Model,
    probability: float | None = None,
) -> np.ndarray:
    """Returns a binary or categorical model based prediction of an image.

    Args:
        image: Image to be predicted.
        model: Model used to predict the image.
        probability: Cutoff value to round model prediction probability.

    Returns:
        List of coordinates [r, c].
    """
    # Normalisation and padding
    image = normalize_image(image)
    pad_bottom = next_power(image.shape[0], 2) - image.shape[0]
    pad_right = next_power(image.shape[1], 2) - image.shape[1]
    image_pad = np.pad(image, ((0, pad_bottom), (0, pad_right)), "reflect")

    # Predict on image
    pred = model.predict(image_pad[None, ..., None]).squeeze()
    prob = 0.5 if probability is None else probability
    coords = get_coordinate_list(
        pred, image_size=max(image_pad.shape), probability=prob
    )

    # Remove spots in padded part of image
    coords = np.array([coords[..., 0], coords[..., 1]])
    coords = np.delete(
        coords,
        np.where((coords[0] > image.shape[0]) | (coords[1] > image.shape[1])),
        axis=1,
    )
    coords = coords.T  # Transposition to save as rows

    # Add third, probability containing column
    if probability is not None:
        probs = get_probabilities(pred, coords, image_size=max(image_pad.shape))
        probs = np.expand_dims(probs, axis=-1)
        coords = np.append(coords, probs, axis=-1)

    return coords


def get_probabilities(
    matrix: np.ndarray, coordinates: np.ndarray, image_size: int = 512
) -> np.ndarray:
    """Find prediction probability given the matrix and coordinates.

    Args:
        matrix: Matrix representation of spot coordinates.
        coordinates: Coordinates at which the probability should be determined.
        image_size: Default image size the grid was layed on.

    Returns:
        Array with all probabilities matching the coordinates.
    """
    matrix_size = max(matrix.shape)
    cell_size = image_size // matrix_size
    nrow = ncol = math.ceil(image_size / cell_size)

    probabilities = []
    for r, c in coordinates:
        # Position of cell coordinate in prediction matrix
        cell_r = min(nrow - 1, int(np.floor(r)) // cell_size)
        cell_c = min(ncol - 1, int(np.floor(c)) // cell_size)

        probabilities.append(matrix[cell_r, cell_c, 0])
    return np.array(probabilities)


def _undo_rotation(coords: np.ndarray, k: int, shape: tuple[int, int]) -> np.ndarray:
    """Reverse a np.rot90(image, k) on coordinates (r, c)."""
    h, w = shape
    r, c = coords[:, 0], coords[:, 1]
    if k == 0:
        return coords
    elif k == 1:  # 90° CCW: (r,c) -> (c, h-1-r)
        return np.column_stack([c, h - 1 - r])
    elif k == 2:  # 180°: (r,c) -> (h-1-r, w-1-c)
        return np.column_stack([h - 1 - r, w - 1 - c])
    elif k == 3:  # 270° CCW: (r,c) -> (w-1-c, r)
        return np.column_stack([w - 1 - c, r])
    return coords


def _undo_flip(coords: np.ndarray, axis: int, shape: tuple[int, int]) -> np.ndarray:
    """Reverse a np.flip(image, axis) on coordinates (r, c)."""
    out = coords.copy()
    if axis == 0:
        out[:, 0] = shape[0] - 1 - out[:, 0]
    else:
        out[:, 1] = shape[1] - 1 - out[:, 1]
    return out


def _merge_predictions(
    all_coords: list[np.ndarray], merge_radius: float = 2.0
) -> np.ndarray:
    """Merge multiple coordinate predictions by averaging nearby detections."""
    if not all_coords or all(len(c) == 0 for c in all_coords):
        return np.empty((0, 2))

    from scipy.spatial import cKDTree

    stacked = np.vstack([c for c in all_coords if len(c) > 0])
    if len(stacked) == 0:
        return np.empty((0, 2))

    tree = cKDTree(stacked)
    used = np.zeros(len(stacked), dtype=bool)
    merged = []

    for i in range(len(stacked)):
        if used[i]:
            continue
        neighbors = tree.query_ball_point(stacked[i], merge_radius)
        cluster = [j for j in neighbors if not used[j]]
        if not cluster:
            continue
        merged.append(stacked[cluster].mean(axis=0))
        used[cluster] = True

    return np.array(merged) if merged else np.empty((0, 2))


def predict_tta(
    image: np.ndarray,
    model: keras.Model,
    probability: float = 0.5,
    merge_radius: float = 2.0,
) -> np.ndarray:
    """Predict with test-time augmentation (4x rotation + 2x flip, averaged).

    Args:
        image: Image to be predicted.
        model: Model used to predict the image.
        probability: Cutoff value for detection probability.
        merge_radius: Radius in pixels for merging nearby detections.

    Returns:
        Merged list of coordinates [r, c].
    """
    shape = image.shape[:2]
    all_preds = []

    # 4 rotations
    for k in range(4):
        img_rot = np.rot90(image, k)
        coords = predict(img_rot, model, probability=None)
        if len(coords) > 0:
            coords = coords[:, :2]  # drop probability column if present
            rot_shape = img_rot.shape[:2]
            coords = _undo_rotation(coords, k, rot_shape)
            all_preds.append(coords)

    # 2 flips
    for axis in [0, 1]:
        img_flip = np.flip(image, axis)
        coords = predict(img_flip, model, probability=None)
        if len(coords) > 0:
            coords = coords[:, :2]
            coords = _undo_flip(coords, axis, shape)
            all_preds.append(coords)

    return _merge_predictions(all_preds, merge_radius)


def get_intensities(
    image: np.ndarray, coordinate_list: np.ndarray, radius: int, method: str = "sum",
) -> np.ndarray:
    """Finds integrated intensities in a radius around each coordinate.

    Args:
        image: Input image with pixel values.
        coordinate_list: List of r, c coordinates in shape (n, 2).
        radius: Radius of kernel to determine intensities.
        method: How the integrated intensity should be calculated
            [options: sum, mean, std].

    Returns:
        Array with all integrated intensities.
    """
    kernel = skimage.morphology.disk(radius)

    for r, c in coordinate_list:
        if not all([isinstance(i, float) for i in [r, c]]):
            print(r, c)

    intensities = np.zeros((len(coordinate_list), 1))
    for idx, (r, c) in enumerate(np.round(coordinate_list).astype(int)):
        # Selection with indexes will be truncated to the max index possible automatically
        area = (
            image[
                max(r - radius, 0) : r + radius + 1,
                max(c - radius, 0) : c + radius + 1,
            ]
            * kernel[
                max(radius - r, 0) : radius + image.shape[0] - r,
                max(radius - c, 0) : radius + image.shape[1] - c,
            ]
        )
        if method == "sum":
            intensities[idx] = np.sum(area)
        elif method == "mean":
            intensities[idx] = np.mean(area)
        elif method == "std":
            intensities[idx] = np.std(area)
        else:
            options = ["sum", "mean", "std"]
            raise ValueError(f'Method must be one of "{options}". {method} is not.')

    return intensities

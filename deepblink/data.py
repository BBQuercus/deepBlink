"""List of functions to handle data including converting matrices <-> coordinates."""


import math
import operator
from typing import Tuple

import numpy as np


def next_power(x: int, k: int = 2) -> int:
    """Calculate x's next higher power of k."""
    y, power = 0, 1
    while y < x:
        y = k ** power
        power += 1
    return y


def next_multiple(x: int, k: int = 512) -> int:
    """Calculate x's closest higher multiple of base k."""
    if x % k:
        x = x + (k - x % k)
    return x


def random_cropping(
    image: np.ndarray, mask: np.ndarray, cell_size: int, crop_size: int = 256
) -> np.ndarray:
    """Randomly crop an image and mask to size crop_size.

    Args:
        image: Image to be cropped.
        mask: Mask to be cropped.
        cell_size: size of cell used to calculate F1 score
        crop_size: Size to crop image and mask (both dimensions).

    Returns:
        crop_image, crop_mask: Cropped image and mask respectively with shape (crop_size, crop_size).
    """
    if not all(isinstance(i, np.ndarray) for i in [image, mask]):
        raise TypeError(
            f"image, mask must be np.ndarray but is {type(image), type(mask)}."
        )
    if not all(isinstance(i, int) for i in [crop_size, cell_size]):
        raise TypeError(
            f"crop_size, cell_size must be an int but is {type(crop_size), type(cell_size)}."
        )
    if crop_size == 0:
        raise ValueError("crop_size must be larger than 0.")
    if not all(image.shape[i] >= crop_size for i in range(2)):
        raise ValueError("crop_size must be smaller than image_size.")
    if crop_size % cell_size > 0:
        raise ValueError("Crop size must be a multiple of cell_size.")

    start_dim = [0, 0]
    if image.shape[0] > crop_size:
        start_dim[0] = int(
            np.floor(
                np.random.randint(low=0, high=image.shape[0] - crop_size) / cell_size
            )
            * cell_size
        )
    if image.shape[1] > crop_size:
        start_dim[1] = int(
            np.floor(
                np.random.randint(low=0, high=image.shape[1] - crop_size) / cell_size
            )
            * cell_size
        )

    cropped_image = image[
        start_dim[0] : (start_dim[0] + crop_size),
        start_dim[1] : (start_dim[1] + crop_size),
    ]
    cropped_mask = mask[
        int(start_dim[0] / cell_size) : int((start_dim[0] + crop_size) / cell_size),
        int(start_dim[1] / cell_size) : int((start_dim[1] + crop_size) / cell_size),
        :,
    ]

    return cropped_image, cropped_mask


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Normalizes images based on bit depth.

    Args:
        images: Input image with uint8 or uint16 formatting.

    Returns:
        Image normalized to 0-1 as float32.
    """
    if images.dtype == np.uint8:
        return (images / 255).astype(np.float32)
    if images.dtype == np.uint16:
        return (images / 65535).astype(np.float32)

    return images


def get_coordinate_list(matrix: np.ndarray, size_image: int = 512) -> np.ndarray:
    """Convert the prediction matrix into a list of coordinates.

    NOTE - if plotting with plt.scatter, x and y must be reversed!

    Args:
        matrix: Matrix representation of spot coordinates.
        size_image: Default image size the grid was layed on.

    Returns:
        Array of x, y coordinates with the shape (n, 2).
    """
    if not matrix.ndim == 3:
        raise ValueError("Matrix must have a shape of (x, y, 3).")
    if not matrix.shape[2] == 3:
        raise ValueError("Matrix must a depth of 3.")
    if not matrix.shape[0] == matrix.shape[1] and not matrix.shape[0] >= 1:
        raise ValueError("Matrix must have equal length >= 1 of x, y.")

    size_grid = matrix.shape[0]
    size_gridcell = size_image // size_grid
    coords_x = []
    coords_y = []

    # Top left coordinates of every cell
    grid = np.array([x * size_gridcell for x in range(size_grid)])

    matrix_x, matrix_y = np.asarray(matrix[..., 0] > 0.5).nonzero()
    for x, y in zip(matrix_x, matrix_y):

        grid_x = grid[x]
        grid_y = grid[y]
        spot_x = matrix[x, y, 1]
        spot_y = matrix[x, y, 2]

        coord_abs = get_absolute_coordinates(
            coord_spot=(spot_x, spot_y),
            coord_cell=(grid_x, grid_y),
            size_gridcell=size_gridcell,
        )

        coords_x.append(coord_abs[0])
        coords_y.append(coord_abs[1])

    return np.array([coords_y, coords_x]).T


def get_absolute_coordinates(
    coord_spot: Tuple[np.float32, np.float32],
    coord_cell: Tuple[np.float32, np.float32],
    size_gridcell: int = 8,
) -> Tuple[np.float32, np.float32]:
    """Return the absolute image coordinates from relative cell coordinates.

    Args:
        coord_spot: Relative spot coordinate in format (x, y).
        coord_cell: Top-left coordinate of the cell.
        size_gridcell: Size of one cell in a grid.

    Returns:
        Absolute coordinate.
    """
    assert len(coord_spot) == 2 and len(coord_cell) == 2

    coord_rel = tuple(map(lambda x: x * size_gridcell, coord_spot))
    coord_abs = tuple(map(operator.add, coord_cell, coord_rel))
    # coord_abs = tuple(map(lambda x: int(x), coord_abs))
    return coord_abs  # type: ignore


def get_prediction_matrix(
    spot_coord: np.ndarray, size: int, cell_size: int, size_y: int = None
) -> np.ndarray:
    """Return np.ndarray of shape (n, n, 3): p, x, y format for each cell.

    Args:
        spot_coord: List of coordinates in x, y format with shape (n, 2).
        size: size of the image from which List of coordinates are extracted.
        cell_size: size of cell used to calculate F1 score, precision and recall.
        size_y: if not provided, it assumes it is squared image, otherwise the second shape of image

    Returns:
        The prediction matrix as numpy array of shape (n, n, 3): p, x, y format for each cell.
    """
    if not all(isinstance(i, int) for i in (size, cell_size)):
        raise TypeError(
            f"size and cell_size must be int, but are {type(size), type(cell_size)}."
        )

    nrow = math.ceil(size / cell_size)
    ncol = nrow
    if size_y is not None:
        ncol = math.ceil(size_y / cell_size)

    pred = np.zeros((nrow, ncol, 3))
    for nspot in range(len(spot_coord)):
        i = int(np.floor(spot_coord[nspot, 0])) // cell_size
        j = int(np.floor(spot_coord[nspot, 1])) // cell_size
        rel_x = (spot_coord[nspot, 0] - i * cell_size) / cell_size
        rel_y = (spot_coord[nspot, 1] - j * cell_size) / cell_size
        pred[i, j, 0] = 1
        pred[i, j, 1] = rel_x
        pred[i, j, 2] = rel_y

    return pred

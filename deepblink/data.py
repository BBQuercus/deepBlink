"""List of functions to handle data including converting matrices <-> coordinates."""


from typing import Tuple
import math
import operator

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
    if np.nanmax(images) != 0:
        return (images / np.nanmax(images)).astype(np.float32)

    return images


def get_coordinate_list(matrix: np.ndarray, image_size: int = 512) -> np.ndarray:
    """Convert the prediction matrix into a list of coordinates.

    NOTE - plt.scatter uses the x, y system. Therefore any plots
    must be inverted by assigning x=c, y=r!

    Args:
        matrix: Matrix representation of spot coordinates.
        image_size: Default image size the grid was layed on.

    Returns:
        Array of r, c coordinates with the shape (n, 2).
    """
    if not matrix.ndim == 3:
        raise ValueError("Matrix must have a shape of (r, c, 3).")
    if not matrix.shape[2] == 3:
        raise ValueError("Matrix must a depth of 3.")
    if not matrix.shape[0] == matrix.shape[1] and not matrix.shape[0] >= 1:
        raise ValueError("Matrix must have equal length >= 1 of r, c.")

    size_grid = matrix.shape[0]
    cell_size = image_size // size_grid
    coords_r = []
    coords_c = []

    # Top left coordinates of every cell
    grid = np.array([c * cell_size for c in range(size_grid)])

    matrix_r, matrix_c = np.round(matrix[..., 0]).nonzero()
    for r, c in zip(matrix_r, matrix_c):

        grid_r = grid[r]
        grid_c = grid[c]
        spot_r = matrix[r, c, 1]
        spot_c = matrix[r, c, 2]

        coord_abs = absolute_coordinate(
            coord_spot=(spot_r, spot_c),
            coord_cell=(grid_r, grid_c),
            cell_size=cell_size,
        )

        coords_r.append(coord_abs[0])
        coords_c.append(coord_abs[1])

    return np.array([coords_r, coords_c]).T


def absolute_coordinate(
    coord_spot: Tuple[np.float32, np.float32],
    coord_cell: Tuple[np.float32, np.float32],
    cell_size: int = 4,
) -> Tuple[np.float32, np.float32]:
    """Return the absolute image coordinate from a relative cell coordinate.

    Args:
        coord_spot: Relative spot coordinate in format (r, c).
        coord_cell: Top-left coordinate of the cell.
        cell_size: Size of one cell in a grid.

    Returns:
        Absolute coordinate.
    """
    if not len(coord_spot) == len(coord_cell) == 2:
        raise ValueError(
            f"coord_spot, coord_cell must have format (r, c). Lengths are {len(coord_spot), len(coord_cell)} resp."
        )

    coord_rel = tuple(map(lambda x: x * cell_size, coord_spot))
    coord_abs = tuple(map(operator.add, coord_cell, coord_rel))
    return coord_abs  # type: ignore


def get_prediction_matrix(
    coords: np.ndarray, image_size: int, cell_size: int = 4, size_c: int = None
) -> np.ndarray:
    """Return np.ndarray of shape (n, n, 3): p, r, c format for each cell.

    Args:
        coords: List of coordinates in r, c format with shape (n, 2).
        image_size: Size of the image from which List of coordinates are extracted.
        cell_size: Size of one grid cell inside the matrix. A cell_size of 2 means that one
            cell corresponds to 2 pixels in the original image.
        size_c: If empty, assumes a squared image. Else the length of the r axis.

    Returns:
        The prediction matrix as numpy array of shape (n, n, 3): p, r, c format for each cell.
    """
    nrow = ncol = math.ceil(image_size / cell_size)
    if size_c is not None:
        ncol = math.ceil(size_c / cell_size)

    prediction_matrix = np.zeros((nrow, ncol, 3))
    for r, c in coords:
        # Position of cell coordinate in prediction matrix
        cell_r = int(np.floor(r)) // cell_size
        cell_c = int(np.floor(c)) // cell_size

        # Relative position within cell
        relative_r = (r - cell_r * cell_size) / cell_size
        relative_c = (c - cell_c * cell_size) / cell_size

        # Assign values along prediction matrix dimension 3
        prediction_matrix[cell_r, cell_c] = 1, relative_r, relative_c

    return prediction_matrix

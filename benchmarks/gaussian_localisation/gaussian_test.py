import argparse
import dask
import glob
import re
import sys
import os

import deepblink as pink
import numpy as np
import scipy.optimize as opt

sys.path.append("../")
from util import delayed_results

EPS = 1e-4


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets")
    parser.add_argument("--models")
    args = parser.parse_args()
    return args


@dask.delayed
def delayed_normalize_pink(image):
    return pink.data.normalize_image(image)


def gauss_2d(xy, amplitude, x0, y0, sigma_xy, offset):
    """2D gaussian."""
    x, y = xy
    x0 = float(x0)
    y0 = float(y0)
    gauss = offset + amplitude * np.exp(
        -(
            ((x - x0) ** (2) / (2 * sigma_xy ** (2)))
            + ((y - y0) ** (2) / (2 * sigma_xy ** (2)))
        )
    )
    return gauss


def gauss_single_spot(
    image: np.ndarray, c_coord: float, r_coord: float, crop_size: int
):
    """Gaussian prediction on a single crop centred on spot."""
    start_dim1 = np.max([int(np.round(r_coord - crop_size // 2)), 0])
    if start_dim1 < len(image) - crop_size:
        end_dim1 = start_dim1 + crop_size
    else:
        start_dim1 = len(image) - crop_size
        end_dim1 = len(image)

    start_dim2 = np.max([int(np.round(c_coord - crop_size // 2)), 0])
    if start_dim2 < len(image) - crop_size:
        end_dim2 = start_dim2 + crop_size
    else:
        start_dim2 = len(image) - crop_size
        end_dim2 = len(image)

    crop = image[start_dim1:end_dim1, start_dim2:end_dim2]

    x = np.arange(0, crop.shape[1], 1)
    y = np.arange(0, crop.shape[0], 1)
    xx, yy = np.meshgrid(x, y)

    # Guess intial parameters
    # x0, y0 center of gaussian
    # sigma: standard deviation of the gaussian
    # amplitude_max: height of gaussian
    x0 = int(crop.shape[0] // 2)  # Middle of the crop
    y0 = int(crop.shape[1] // 2)  # Middle of the crop
    sigma = max(*crop.shape) * 0.1  # 10% of the crop
    amplitude_max = max(np.max(crop) / 2, np.min(crop))  # Maximum value of the crop

    initial_guess = [amplitude_max, x0, y0, sigma, 0]

    # lower: lower bound for parameter search space
    # upper: upper bound for parameter search space
    lower = [np.min(crop), 0, 0, 0, -np.inf]
    upper = [
        np.max(crop) + EPS,
        crop_size,
        crop_size,
        np.inf,
        np.inf,
    ]
    bounds = [lower, upper]
    try:
        popt, _ = opt.curve_fit(
            gauss_2d,
            (xx.ravel(), yy.ravel()),
            crop.ravel(),
            p0=initial_guess,
            bounds=bounds,
        )
    except RuntimeError:
        return r_coord, c_coord

    x0 = popt[1] + start_dim2
    y0 = popt[2] + start_dim1

    # if predicted spot is out of the border of the image
    if x0 >= image.shape[1] or y0 >= image.shape[0]:
        return r_coord, c_coord

    return y0, x0


@dask.delayed
def gauss_single_image(image: np.ndarray, mask: np.ndarray, crop_size: int = 4):
    """Gaussian localization on a single image using initialized on deepblink output (mask)."""
    coord_list = pink.data.get_coordinate_list(mask.squeeze(), 512)

    prediction_coord = []
    for i in range(len(coord_list)):
        r_coord = min(len(image) - EPS, coord_list[i, 0])
        c_coord = min(len(image) - EPS, coord_list[i, 1])

        prediction_coord.append(gauss_single_spot(image, c_coord, r_coord, crop_size))

    return np.array(prediction_coord)


if __name__ == "__main__":
    args = _parse_args()
    models = sorted(glob.glob(f"{args.models}/*.h5"))
    datasets = sorted(glob.glob(f"{args.datasets}/*.npz"))
    md_deepblink = [
        (re.search(r"deepblink_([a-z]+)\.", m)[1], m, d)
        for m, d in zip(models, datasets)
    ]

    results_deepblink = delayed_results(
        model_dataset=md_deepblink,
        model_loader=pink.io.load_model,
        delayed_normalize=delayed_normalize_pink,
        delayed_coordinates=gauss_single_image,
    )
    results_deepblink.to_csv("deepblink_gaussian.csv")

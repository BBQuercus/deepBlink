import argparse
import dask
import glob
import re
import sys

import deepblink as pink
import tensorflow as tf

sys.path.append("../")
from util import delayed_results


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets")
    parser.add_argument("--models")
    args = parser.parse_args()
    return args


@dask.delayed
def delayed_normalize_pink(image):
    return pink.data.normalize_image(image)


@dask.delayed
def delayed_coordinates_pink(mask):
    return pink.data.get_coordinate_list(mask.squeeze(), image_size=512)


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
        delayed_coordinates=delayed_coordinates_pink,
    )
    results_deepblink.to_csv("deepblink.csv")

import argparse
import glob
import re
import sys

import numpy
import skimage.util

from detnet_train import detnet

sys.path.append("../")
from util import delayed_results
from util import delayed_normalize
from util import delayed_coordinates


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets")
    parser.add_argument("--models")
    args = parser.parse_args()
    return args


def model_loader_detnet(fname):
    """Load spotlearn model with weights."""
    alpha = float(re.search(r"_(0\.\d{4})\.", fname)[1])
    model = detnet(image_size=512, alpha=alpha)
    model.load_weights(fname)
    return model


if __name__ == "__main__":
    args = _parse_args()
    models = sorted(glob.glob(f"{args.models}/*.h5"))
    datasets = sorted(glob.glob(f"{args.datasets}/*.npz"))
    md_detnet = [
        (re.search(r"detnet_([a-z]+)_", m)[1], m, d) for m, d in zip(models, datasets)
    ]
    results_detnet = delayed_results(
        model_dataset=md_detnet,
        model_loader=model_loader_detnet,
        delayed_normalize=delayed_normalize,
        delayed_coordinates=delayed_coordinates,
    )
    results_detnet.to_csv("detnet.csv")

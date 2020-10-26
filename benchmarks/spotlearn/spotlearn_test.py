import argparse
import glob
import re
import sys

from spotlearn_train import get_unet_short_dropout

sys.path.append("../")
from util import delayed_results
from util import delayed_coordinates
from util import delayed_normalize


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets")
    parser.add_argument("--models")
    args = parser.parse_args()
    return args


def model_loader_spotlearn(fname):
    """Load spotlearn model with weights."""
    model = get_unet_short_dropout(img_size=512)
    model.load_weights(fname)
    return model


if __name__ == "__main__":
    args = _parse_args()
    models = sorted(glob.glob(f"{args.models}/*.h5"))
    datasets = sorted(glob.glob(f"{args.datasets}/*.npz"))
    md_spotlearn = [
        (re.search(r"spotlearn_([a-z]+)_", m)[1], m, d)
        for m, d in zip(models, datasets)
    ]
    results_spotlearn = delayed_results(
        model_dataset=md_spotlearn,
        model_loader=model_loader_spotlearn,
        delayed_normalize=delayed_normalize,
        delayed_coordinates=delayed_coordinates,
    )
    results_spotlearn.to_csv("spotlearn.csv")

"""Test run of detnet on given dataset."""
import re
import sys
import textwrap

import numpy as np

sys.path.append("../")
from detnet_train import detnet
from util import _parse_args
from util import get_coordinates
from util import run_test


def model_loader(fname, img_size):
    """Load spotlearn model with weights."""
    alpha = float(re.search(r"\d{8}_(\d\.\d+)\.h5", fname)[1])
    model = detnet(img_size, alpha)
    model.load_weights(fname)
    return model


def normalize_fn(image):
    """Normalize a single input image."""
    return image.astype(np.float32)


def main():
    # Argparse
    args = _parse_args(test=True)
    dataset = args.dataset
    output = args.output
    fname_model = args.model

    # Evaluation
    f1_mean, f1_std, rmse_mean, rmse_std = run_test(
        benchmark="detnet",
        dataset=dataset,
        output=output,
        fname_model=fname_model,
        img_size=512,
        model_loader=model_loader,
        normalize_fn=normalize_fn,
        coordinate_loader=get_coordinates,
    )

    print(
        textwrap.dedent(
            f"""Metrics on test set:
        * F1 score: {f1_mean} ± {f1_std}
        * Mean euclidean distance: {rmse_mean} ± {rmse_std}"""
        )
    )


if __name__ == "__main__":
    main()

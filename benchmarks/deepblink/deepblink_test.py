"""Test run of deepblink on given dataset."""
import sys
import textwrap

import deepblink as pink
import numpy as np
import tensorflow as tf

sys.path.append("../")
from util import _parse_args
from util import run_test


def model_loader(fname, img_size):
    """Load deepblink model with weights."""
    model = tf.keras.models.load_model(
        fname,
        custom_objects={
            "f1_score": pink.losses.f1_score,
            "rmse": pink.losses.rmse,
            "combined_f1_rmse": pink.losses.combined_f1_rmse,
            "combined_bce_rmse": pink.losses.combined_bce_rmse,
        },
    )
    return model


def main():
    # Argparse
    args = _parse_args(test=True)
    dataset = args.dataset
    output = args.output
    fname_model = args.model

    # Evaluation
    f1_mean, f1_std, rmse_mean, rmse_std = run_test(
        benchmark="deepblink",
        dataset=dataset,
        output=output,
        fname_model=fname_model,
        img_size=512,
        model_loader=model_loader,
        normalize_fn=pink.data.normalize_image,
        coordinate_loader=pink.data.get_coordinate_list,
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

"""Training step of trackmate after the prediction in FIJI."""
import argparse
import glob
import os
import re
import sys
import textwrap
import warnings

from dask.distributed import Client
import dask
import dask.dataframe as dd
import deepblink as pink
import numpy as np
import pandas as pd
import scipy.spatial

sys.path.append("../")
from util import _parse_args_fiji


DTYPES = {
    "radius": np.float16,
    "threshold": np.float16,
    "x": np.float16,
    "y": np.float16,
    "q": np.float64,
}


def get_thresholds(files: list, num: int = 20) -> list:
    """Compute all relative thresholds.

    Because quality scores depend on local pixel intensities and contrast ratios,
    they can vary wildly. To counteract this thresholds will be calculated as
    follows:
        * Normalize the quality score by each image predictions median
        * Pool all scores and based on global values...
        * Extract 20 values, evenly distributed between percentiles 1 and 99.

    This is calculated in raw pandas because of strange behaviour when using
    dask dataframes. PRs solving this issue are welcome.

    Args:
        files: A list of all csv files.
        num: Number of percentile based thresholds.
    """
    warnings.warn(
        "This step is very memory intensive. Make sure you have RAM to fit all csv files.",
        ResourceWarning,
    )

    df = pd.concat([pd.read_csv(f, dtype=DTYPES) for f in files])
    df["norm_q"] = df["q"] - df.groupby(["fname", "radius"])["q"].transform("median")

    quantiles = np.linspace(0.01, 0.99, num=num)
    return [df["norm_q"].quantile(q) for q in quantiles]


@dask.delayed(nout=4)
def load_prediction(file):
    """Load the prediction csv file."""
    df = pd.read_csv(file, dtype=DTYPES)

    # Metadata
    fname = pink.io.basename(df["fname"][0])
    detector = df["detector"][0]
    radius = df["radius"][0]

    # Normalized quality scores
    df["norm_q"] = df["q"] - df["q"].median()
    pred_all = df[["y", "x", "norm_q"]].to_numpy()

    return fname, detector, radius, pred_all


@dask.delayed
def load_true(basedir, fname):
    """Load the as npy file saved ground truth."""
    return np.load(os.path.join(basedir, "train_labels", f"{fname}.npy"))


@dask.delayed
def process(fname, detector, radius, pred_all, true, thresholds):
    """Processes a single prediction / true set returning a DataFrame with all metrics."""

    scores = []

    for threshold in thresholds:
        pred = pred_all[:, :2][(pred_all[:, 2] >= threshold)]

        if (length := len(pred)) != 0:
            matrix = scipy.spatial.distance.cdist(pred, true, metric="euclidean")
            pred_true_r, pred_true_c = pink.metrics.linear_sum_assignment(
                matrix, cutoff=5
            )
            true_pred_r, true_pred_c = pink.metrics.linear_sum_assignment(
                matrix.T, cutoff=5
            )

            true_positive = len(true_pred_r)
            false_negative = len(true) - len(true_pred_r)
            false_positive = len(pred) - len(pred_true_r)
            recall = true_positive / (true_positive + false_negative + 1e-10)
            precision = true_positive / (true_positive + false_positive + 1e-10)
            f1_value = (2 * precision * recall) / (precision + recall + 1e-10)
        else:
            f1_value = 0.0

        scores.append([fname, detector, radius, threshold, length, f1_value])

    df = pd.DataFrame(
        scores,
        columns=["fname", "detector", "radius", "threshold", "length", "f1_score"],
    )
    return df


@dask.delayed
def save(basedir, fname, df):
    """Save the output of one run as csv file."""
    outname = os.path.join(basedir, "train_processed", f"{fname}.csv")
    df.to_csv(outname, index=False)
    return outname


def get_max_scores(basedir: str):
    """Compute the optimal parameters for the dataset."""
    files = glob.glob(os.path.join(basedir, "train_processed", "*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files])

    df_f1 = (
        df.groupby(["detector", "radius", "threshold"])["f1_score"]
        .mean()
        .reset_index()
        .nlargest(1, "f1_score")
        .reset_index()
    )
    return df_f1


def main():
    """Evaluation loop finding the best trackmate parameters in three steps.

    1. Find set of thresholds based on median-normalized quantiles
    2. Process all files distributed outputting the F1 score across thresholds
    3. Meaning across images and determination of best parameters
    """
    args = _parse_args_fiji()
    basedir = args.basedir

    # Start dask client with default: localhost:8787
    client = Client()
    print("Dask client started.")

    # Calculate thresholds
    files = glob.glob(os.path.join(basedir, "train_predictions", "*.csv"))
    thresholds = get_thresholds(files)
    print("Thresholds calculated.")

    # Process files
    def f(files):
        """Processes all files by calling delayed functions for each step."""
        results = []

        for file in files:
            fname, detector, radius, pred_all = load_prediction(file)
            true = load_true(basedir, fname)
            df = process(fname, detector, radius, pred_all, true, thresholds)
            result = save(basedir, fname, df)
            results.append(result)

        return results

    # Compute files in batch
    length = len(files)
    batch_size = 100
    for idx in range(0, length, batch_size):
        dask.compute(f(files[idx : min(idx + batch_size, length)]))
    print("Files processed.")

    # Compute maximal scores
    df_max = get_max_scores(basedir)
    print(
        textwrap.dedent(
            f"""Optimal metrics found are:
        * F1 score: {df_max["f1_score"][0]}
        * Detector: {df_max["detector"][0]}
        * Radius: {df_max["radius"][0]}
        * Threshold: {df_max["threshold"][0]}"""
        )
    )

    client.shutdown()


if __name__ == "__main__":
    main()

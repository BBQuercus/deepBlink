"""Test run of trackmate after prediction in FIJI."""
import glob
import os
import sys
import textwrap

from dask.distributed import Client
import dask
import deepblink as pink
import numpy as np
import pandas as pd

sys.path.append("../")
from util import _parse_args_fiji
from util import compute_metrics


DTYPES = {
    "radius": np.float16,
    "threshold": np.float16,
    "x": np.float16,
    "y": np.float16,
    "q": np.float64,
}


@dask.delayed(nout=2)
def load_prediction(file, threshold):
    """Load the prediction csv file."""
    df = pd.read_csv(file, dtype=DTYPES)

    # Metadata
    fname = pink.io.basename(df["fname"][0])

    # Normalized quality scores
    df["norm_q"] = df["q"] - df["q"].median()
    pred_all = df[["y", "x", "norm_q"]].to_numpy()
    pred = pred_all[:, :2][(pred_all[:, 2] >= threshold)]

    return fname, pred


@dask.delayed
def load_true(basedir, fname):
    """Load the as npy file saved ground truth."""
    return np.load(os.path.join(basedir, "test_labels", f"{fname}.npy"))


@dask.delayed
def process(fname, pred, true):
    """Processes a single prediction / true set returning a DataFrame with all metrics."""
    df = compute_metrics(pred, true)
    df["fname"] = fname
    return df


@dask.delayed
def save(basedir, fname, df):
    """Save the output of one run as csv file."""
    outname = os.path.join(basedir, "test_processed", f"{fname}.csv")
    df.to_csv(outname)
    return outname


def main():
    """Evaluation loop finding the best trackmate parameters in three steps.

    1. Find set of thresholds based on median-normalized quantiles
    2. Process all files distributed outputting the F1 score across thresholds
    3. Meaning across images and determination of best parameters
    """
    args = _parse_args_fiji(test=True)
    basedir = args.basedir
    threshold = args.threshold

    # Start dask client with default: localhost:8787
    client = Client()
    print("Dask client started.")

    files = glob.glob(os.path.join(basedir, "test_predictions", "*.csv"))

    def f(files):
        """Processes all files by calling delayed functions for each step."""
        results = []

        for file in files:
            fname, pred = load_prediction(file, threshold)
            true = load_true(basedir, fname)
            df = process(fname, pred, true)
            result = save(basedir, fname, df)
            results.append(result)

        return results

    result = dask.compute(f(files))[0]  # Returns tuple
    print("Files processed.")

    df = pd.concat([pd.read_csv(file) for file in result])
    print(
        textwrap.dedent(
            f"""Metrics on test set:
        * F1 score: {df["f1_integral"].mean()} ± {df["f1_integral"].std()}
        * Mean euclidean distance: {df["mean_euclidean"].mean()} ± {df["mean_euclidean"].std()}"""
        )
    )

    client.shutdown()


if __name__ == "__main__":
    main()

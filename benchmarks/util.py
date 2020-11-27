import glob
import os
import re
import sys

import dask
import deepblink as pink
import numpy as np
import pandas as pd
import skimage.measure
import skimage.util

DTYPES = {
    "radius": np.float16,
    "threshold": np.float16,
    "x": np.float16,
    "y": np.float16,
    "q": np.float64,
}


def print_results(results: pd.DataFrame):
    for name in results.name.unique():
        df = results[results["name"] == name]

        p = 4
        f1_m = df[df["cutoff"] == 3]["f1_score"].mean().round(p)
        f1_s = df[df["cutoff"] == 3]["f1_score"].std().round(p)
        f1i_m = df["f1_integral"].mean().round(p)
        f1i_s = df["f1_integral"].std().round(p)
        rmse_m = df["mean_euclidean"].mean().round(p)
        rmse_s = df["mean_euclidean"].std().round(p)

        print(
            f"Evaluation of {name}:\n"
            f"    F1 @3px: {f1_m} ± {f1_s}\n"
            f"    F1 integral @3px: {f1i_m} ± {f1i_s}\n"
            f"    RMSE @3px: {rmse_m} ± {rmse_s}\n"
        )


@dask.delayed
def delayed_normalize(image):
    if image.dtype == float:
        image = skimage.util.img_as_ubyte(image)
    image = image.astype(np.float32)
    image /= 255
    return image


@dask.delayed
def delayed_predict(model, image):
    return model.predict(image[None, ..., None])


@dask.delayed
def delayed_coordinates(mask):
    binary = np.round(mask.squeeze())
    label = skimage.measure.label(binary)
    props = skimage.measure.regionprops(label)
    coords = np.array([p.centroid for p in props])
    return coords


@dask.delayed
def delayed_compute(pred, true, idx):
    df = pink.metrics.compute_metrics(pred, true, mdist=3)
    df["image"] = idx
    return df


def delayed_results(
    model_dataset: tuple,
    model_loader: callable,
    delayed_normalize: callable,
    delayed_coordinates: callable,
):
    """Distributedly compute all results across datasets and model pairs."""
    results = pd.DataFrame()
    for name, fname_model, dataset in model_dataset:
        print(f"Starting evaluation of {name}.")

        # Import data and model
        x_test, y_test = pink.io.load_npz(dataset, test_only=True)
        model = model_loader(fname_model)

        def f(images, trues):
            results = []

            for idx, (image, true) in enumerate(zip(images, trues)):
                image = delayed_normalize(image)
                pred = delayed_predict(model, image)
                try:
                    pred = delayed_coordinates(pred)
                except:
                    pred = delayed_coordinates(image, pred)
                df = delayed_compute(pred, true, idx)
                results.append(df)
            return results

        # Compute results

        result = dask.compute(f(x_test, y_test))[0]  # Returns tuple
        df = pd.concat(result)
        df["name"] = name
        results = results.append(df)
    return results

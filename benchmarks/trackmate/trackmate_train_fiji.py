# @ String basedir

"""Training script for trackmate's FIJI API."""

import csv
import glob
import os

from ij import IJ
from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.detection import LogDetectorFactory
from fiji.plugin.trackmate.detection import DogDetectorFactory


def run_trackmate(
    imp, path_out="./", detector="log", radius=2.5, threshold=0.0, median_filter=False
):
    """Log Trackmate detection run with given parameters.
    Saves spots in a csv file in the given path_out with encoded parameters.

    Args:
        imp: ImagePlus to be processed
        path_out: Output directory to save files.
        detector: Type of detection method. Options are 'log', 'dog'.
        radius: Radius of spots in pixels.
        threshold: Threshold value to filter spots.
        median_filter: True if median_filtering should be used.
    """
    if imp.dimensions[2] != 1:
        raise ValueError(
            "Imp's dimensions must be [n, n, 1] but are " + imp.dimensions[2]
        )

    # Create the model object now
    model = Model()
    model.setLogger(Logger.VOID_LOGGER)

    # Prepare settings object
    settings = Settings()
    settings.setFrom(imp)

    # Configure detector
    settings.detectorFactory = (
        DogDetectorFactory() if detector == "dog" else LogDetectorFactory()
    )
    settings.detectorSettings = {
        "DO_SUBPIXEL_LOCALIZATION": True,
        "RADIUS": radius,
        "TARGET_CHANNEL": 1,
        "THRESHOLD": threshold,
        "DO_MEDIAN_FILTERING": median_filter,
    }

    # Instantiate plugin
    trackmate = TrackMate(model, settings)

    # Process
    # output = trackmate.process()
    output = trackmate.execDetection()
    if not output:
        print("error process")
        return None

    # Get output from a single image
    fname = str(imp.title)
    spots = [["fname", "detector", "radius", "threshold", "median", "x", "y", "q"]]
    for spot in model.spots.iterator(0):
        x = spot.getFeature("POSITION_X")
        y = spot.getFeature("POSITION_Y")
        q = spot.getFeature("QUALITY")
        spots.append([fname, detector, radius, threshold, median_filter, x, y, q])

    print(spots)

    # Save output
    outname = os.path.splitext(os.path.basename(fname))[0] + "_" + str(radius) + ".csv"
    with open(os.path.join(path_out, outname), "wb") as f:
        wr = csv.writer(f)
        for row in spots:
            wr.writerow(row)


def linspace(a, b, n=25):
    """Return N linearly distributed numbers between a and b."""
    if n < 2:
        return b
    diff = (float(b) - a) / (n - 1)
    return [round(diff * i + a, 4) for i in range(n)]


def run_train(outdir, fname):
    """One "training" loop in trackmate.
    
    Open an image in the ImageJ image file format and run trackmate with
    a selection of different parameters.
    """
    imp = IJ.openImage(fname)

    # opts_detector = ["log", "dog"]
    opts_radius = [0.01, 0.1, 0.2, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]

    for r in opts_radius:
        run_trackmate(imp, path_out=outdir, radius=r)

    imp.close()


def main():
    indir = os.path.join(basedir, "train_images")
    outdir = os.path.join(basedir, "train_predictions")

    files = sorted(glob.glob(os.path.join(indir, "*.tif")))
    for file in files:
        print("Running: " + file)
        run_train(outdir, file)


main()

# @String basedir
# @String detector
# @Float radius
# @Boolean median

"""Testing script for trackmate's FIJI API."""

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


# NOTE as FIJI / jython can't import, this function should only be changed
# in conjunction with the same named function in train.py
def run_trackmate(
    imp, path_out="./", detector="log", radius=2.5, threshold=0.0, median_filter=False
):
    """Log Trackmate detection run with given parameters.
    Saves spots in a csv file in the given path_out with encoded parameters.

    Args:
        imp: ImagePlus to be processed
        path_out: Output directory to save files.
        detector: Type of detection method. Options are "log", "dog".
        radius: Radius of spots in pixels.
        threshold: Threshold value to filter spots.
        median_filter: True if median_filtering should be used.
    """
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

    # Save output
    outname = os.path.splitext(os.path.basename(fname))[0] + "_" + str(radius) + ".csv"
    with open(os.path.join(path_out, outname), "wb") as f:
        wr = csv.writer(f)
        for row in spots:
            wr.writerow(row)


def run_test(outdir, fname):
    imp = IJ.openImage(fname)

    run_trackmate(
        imp,
        path_out=os.path.join(basedir, "test_predictions"),
        detector=detector,
        radius=radius,
        threshold=0.0,  # in trackmate_test.py after mean-normalization
        median_filter=median,
    )

    imp.close()


def main():
    indir = os.path.join(basedir, "test_images")
    outdir = os.path.join(basedir, "test_predictions")

    files = sorted(glob.glob(os.path.join(indir, "*.tif")))
    for file in files:
        print("Running: " + file)
        run_test(outdir, file)


main()

"""Dataset preparation for to allow trackmate to access all files easily."""
import os
import sys

import deepblink as pink
import numpy as np
import skimage.util
import tifffile

sys.path.append("../")
from util import _parse_args


def save_images(images, location, basename):
    """Save ImageJ compatible images at location with basename."""
    for n, image in enumerate(images):
        if image.dtype == float:
            image = skimage.util.img_as_ubyte(image)
        tifffile.imsave(f"{location}/{basename}_{n}.tif", image, imagej=True)
    print(f"{n+1} images saved for {basename}.")


def save_labels(labels, location, basename):
    """Save single labels as npy files at location with basename."""
    for n, label in enumerate(labels):
        np.save(f"{location}/{basename}_{n}.npy", label)
    print(f"{n+1} labels saved for {basename}.")


def main():
    args = _parse_args()
    dataset = args.dataset
    output = args.output

    # Make subdirs
    subdirs = [
        "test_images",
        "test_labels",
        "test_predictions",
        "test_processed",
        "train_images",
        "train_labels",
        "train_predictions",
        "train_processed",
    ]
    bname_dataset = os.path.join(output, pink.io.extract_basename(dataset))
    os.makedirs(bname_dataset, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(os.path.join(bname_dataset, subdir), exist_ok=True)

    # Loading
    x_train, y_train, x_valid, y_valid, x_test, y_test = pink.io.load_npz(dataset)

    # Training
    save_images(x_train, os.path.join(bname_dataset, "train_images"), "train")
    save_images(x_valid, os.path.join(bname_dataset, "train_images"), "valid")
    save_labels(y_train, os.path.join(bname_dataset, "train_labels"), "train")
    save_labels(y_valid, os.path.join(bname_dataset, "train_labels"), "valid")

    # Testing
    save_images(x_test, os.path.join(bname_dataset, "test_images"), "test")
    save_labels(y_test, os.path.join(bname_dataset, "test_labels"), "test")


if __name__ == "__main__":
    main()

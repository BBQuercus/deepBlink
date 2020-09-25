"""Test run of spotlearn on given dataset."""
import sys

import numpy as np
import skimage.util

sys.path.append("../")
from spotlearn_train import get_unet_short_dropout
from util import _parse_args
from util import get_coordinates
from util import run_test


def model_loader(fname, img_size):
    """Load spotlearn model with weights."""
    model = get_unet_short_dropout(img_size)
    model.load_weights(fname)
    return model


def normalize_fn(image):
    """Normalize a single input image."""
    if image.dtype == float:
        image = skimage.util.img_as_ubyte(image)
    image = image.astype(np.float32)
    image /= 255
    return image


def main():
    # Argparse
    args = _parse_args(test=True)
    dataset = args.dataset
    output = args.output
    fname_model = args.model

    # Evaluation
    run_test(
        benchmark="spotlearn",
        dataset=dataset,
        output=output,
        fname_model=fname_model,
        img_size=512,
        model_loader=model_loader,
        normalize_fn=normalize_fn,
        coordinate_loader=get_coordinates,
    )


if __name__ == "__main__":
    main()

"""Utilities shared by multiple CLI modules."""
from typing import Tuple, Union
import logging

from ..util import predict_pixel_size


def get_pixel_size(
    pixel_size: Union[float, Tuple[float, float]], image: str, logger: logging.Logger
) -> Tuple[float, float]:
    """Return the pixel size of an image."""
    # Use user-provided pixel size
    if pixel_size is not None:
        logger.debug(f"using provided pixel size {pixel_size}")
        if isinstance(pixel_size, tuple):
            return pixel_size
        return (pixel_size, pixel_size)

    # Try to predict pixel size
    try:
        size = predict_pixel_size(image)
        logger.debug(f"using predicted pixel size {size}")
        return size
    except (ValueError, KeyError, IndexError) as e:
        logger.warning(
            f"\U000026A0 {e} encountered. Pixel size for image {image} could not be predicted."
        )

    logger.debug(f"defaulting pixel size to {size}")
    return (1.0, 1.0)

"""Model utility functions for augmentation."""

from typing import Tuple

import numpy as np


def augment_batch_baseline(
    images: np.ndarray,
    masks: np.ndarray,
    flip_: bool = False,
    illuminate_: bool = False,
    gaussian_noise_: bool = False,
    rotate_: bool = False,
    translate_: bool = False,
    cell_size: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline augmentation function.

    Probability of augmentations is determined in the corresponding functions
    and not in this baseline.

    Args:
        images: Batch of input image to be augmented.
        masks: Batch of corresponding prediction matrix with ground truth values.
        flip_: If True, images might be flipped.
        illuminate_: If True, images might be altered in illumination.
        gaussian_noise_: If True, gaussian noise might be added.
        rotate_: If True, images might be rotated.
        translate_: If True, images might be translated.
        cell_size: Size of one cell in the prediction matrix.
    """
    aug_images = []
    aug_masks = []

    for image, mask in zip(images, masks):
        aug_image = image.copy()
        aug_mask = mask.copy()

        if flip_:
            aug_image, aug_mask = flip(aug_image, aug_mask)
        if illuminate_:
            aug_image, aug_mask = illuminate(aug_image, aug_mask)
        if gaussian_noise_:
            aug_image, aug_mask = gaussian_noise(aug_image, aug_mask)
        if rotate_:
            aug_image, aug_mask = rotate(aug_image, aug_mask)
        if translate_:
            aug_image, aug_mask = translate(aug_image, aug_mask, cell_size=cell_size)

        aug_images.append(aug_image)
        aug_masks.append(aug_mask)

    aug_images = np.array(aug_images, dtype=np.float32)
    aug_masks = np.array(aug_masks, dtype=np.float32)

    return aug_images, aug_masks


def flip(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Augment through horizontal/vertical flipping."""
    rand_flip = np.random.randint(low=0, high=2)

    image = np.flip(image.copy(), rand_flip)
    mask = np.flip(mask.copy(), rand_flip)

    # Horizontal flip / change along x axis
    if rand_flip == 0:
        mask[..., 1] = np.where(mask[..., 0], 1 - mask[..., 1], mask[..., 1])

    # Vertical flip / change along y axis
    if rand_flip == 1:
        mask[..., 2] = np.where(mask[..., 0], 1 - mask[..., 2], mask[..., 2])

    return image, mask


def illuminate(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Augment through changing illumination."""
    rand_illumination = 1 + np.random.uniform(-0.75, 0.75)
    image = image.copy()
    image = np.multiply(image, rand_illumination)
    return image, mask


def gaussian_noise(
    image: np.ndarray, mask: np.ndarray, mean: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Augment through the addition of gaussian noise.

    Args:
        image: Image to be augmented.
        mask: Corresponding prediction matrix with ground truth values.
        mean: Average noise pixel values added. Zero means no net difference occurs.
    """
    sigma = np.random.uniform(0.0001, 0.01)
    noise = np.random.normal(mean, sigma, image.shape)
    image = image.copy()

    def _gaussian_noise(image: np.ndarray) -> np.ndarray:
        """Gaussian noise helper."""
        mask_overflow_upper = image + noise >= 1.0
        mask_overflow_lower = image + noise < 0
        noise[mask_overflow_upper] = 1.0
        noise[mask_overflow_lower] = 0
        image = np.add(image, noise)
        return image

    image = _gaussian_noise(image)

    return image, mask


def rotate(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Augment through rotation."""
    rand_rotate = np.random.randint(low=0, high=4)
    image = image.copy()
    mask = mask.copy()
    for _ in range(rand_rotate):

        image = np.rot90(image)  # rotate image -90 degrees
        mask = np.rot90(mask)  # rotate mask -90 degrees
        x_coord = mask[..., 1].copy()
        y_coord = mask[..., 2].copy()
        mask[..., 1] = 1 - y_coord  # rotation coordinates +90 degrees + translation
        mask[..., 2] = x_coord  # rotation coordinates +90 degrees
        mask[..., 1][mask[..., 0] == 0] = 0
        mask[..., 2][mask[..., 0] == 0] = 0

    return image, mask


def translate(
    image: np.ndarray, mask: np.ndarray, cell_size: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """Augment through translation along all axes.

    Args:
        image: Image to be augmented.
        mask: Corresponding prediction matrix with ground truth values.
        cell_size: Size of one cell in the prediction matrix.
    """
    direction = np.random.choice([0, 1])
    image = image.copy()
    mask = mask.copy()

    shift_mask = np.random.choice(range(len(image) // cell_size))
    shift_image = shift_mask * cell_size

    image = np.roll(image, shift_image, direction)
    mask = np.roll(mask, shift_mask, direction)

    return image, mask

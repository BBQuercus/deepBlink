"""Train detnet on given dataset with automatic hyperparameter search."""
import datetime
import glob
import itertools
import os
import sys
import textwrap

from tensorflow.keras import backend as K
import deepblink as pink
import numpy as np
import pandas as pd
import skimage.util
import tensorflow as tf
import tensorflow_addons as tfa

sys.path.append("../")
from util import _parse_args
from util import compute_metrics
from util import get_coordinates
from util import plot_metrics


def get_seg_maps(arr: np.ndarray, size: int = 512):
    """Convert coordinate list into single-pixel segmentation maps."""
    seg_maps = np.zeros((len(arr), size, size, 1))

    for i, img in enumerate(arr):
        for coord in np.round(img).astype(int):
            seg_maps[i, min(coord[1], size - 1), min(coord[0], size - 1)] = 1

    return seg_maps


def detnet(image_size: int = None, alpha: float = 0.0) -> tf.keras.models.Model:
    """Return the DetNet architecture.

    Args:
        image_size: Size of input images with format (image_size, image_size).
        alpha: Balance parameter in the sigmoid layer.
    """

    def custom_sigmoid(x):
        return K.sigmoid(x - alpha)

    def conv_norm(inputs, filters):
        conv = tf.keras.layers.Conv2D(filters, **OPTS_CONV)(inputs)
        norm = tfa.layers.InstanceNormalization()(conv)
        return norm

    OPTS_CONV = {"kernel_size": (3, 3), "padding": "same", "activation": "relu"}

    inputs = tf.keras.Input((image_size, image_size, 1))

    conv1 = conv_norm(inputs, 16)
    skip1 = conv_norm(conv1, 16)
    skip2 = conv_norm(skip1, 16)
    add1 = tf.keras.layers.Add()([conv1, skip2])
    max1 = tf.keras.layers.MaxPool2D()(add1)

    conv2 = conv_norm(max1, 32)
    skip3 = conv_norm(conv2, 32)
    skip4 = conv_norm(skip3, 32)
    add2 = tf.keras.layers.Add()([conv2, skip4])
    max2 = tf.keras.layers.MaxPool2D()(add2)

    conv3 = conv_norm(max2, 64)
    skip5 = conv_norm(conv3, 64)
    skip6 = conv_norm(skip5, 64)
    add3 = tf.keras.layers.Add()([conv3, skip6])

    up1 = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")(add3)
    conv4 = conv_norm(up1, 32)
    skip7 = conv_norm(conv4, 32)
    skip8 = conv_norm(skip7, 32)

    up2 = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")(skip8)
    conv5 = conv_norm(up2, 16)
    skip9 = conv_norm(conv5, 16)
    skip10 = conv_norm(skip9, 16)

    logit = tf.keras.layers.Conv2D(
        1, (1, 1), activation=tf.keras.layers.Activation(custom_sigmoid)
    )(skip10)
    outputs = logit

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


def soft_dice(true, pred, eps: float = 1e-7):
    """Compute soft dice on a batch of predictions.

    The soft dice is defined as follows with N denoting all pixels in an image.
    According to [Wollmann, 2019](https://ieeexplore.ieee.org/abstract/document/8759234/),
    calculating the Dice loss over all N pixels in a batch instead of averaging the Dice
    loss over the single images improves training stability.
    """

    # [b, h, w, 1] -> [b*h*w*1]
    true = K.flatten(true)
    pred = K.flatten(pred)

    # [sum(b), h*w]
    multed = K.sum(true * pred)
    summed = K.sum(true + pred)
    dices = 2.0 * ((multed + eps) / (summed + eps))

    return -dices


class DataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for keras' fitting."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        image_size: int = 512,
        shuffle: bool = True,
    ):
        self.batch_size = batch_size
        self.images = labels
        self.labels = labels
        self.image_size = image_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        x, y = self.__data_generation(indexes)
        return x, y

    def on_epoch_end(self):
        """Update of indexes after each epoch."""
        self.indexes = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples."""
        x = np.take(self.images, indexes, axis=0)
        y = np.take(self.labels, indexes, axis=0)
        return x, y


def f1_score(y_true, y_pred):
    """F1 score metric on a batch of images.
    
    NOTE - this is not the f1_integral used in evaluation but simply
    serves as metric along the way.
    """

    def __recall_score(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def __precision_score(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = __precision_score(y_true, y_pred)
    recall = __recall_score(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def rmse_score(y_true, y_pred):
    """RSME score metric on batch of images."""
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def run_train_eval(
    dataset: str, output: str, img_size: int, alpha: float, batch_size: int = 8
) -> None:
    """Main training loop with one set of hyperparameters.

    Args:
        img_size: See function detnet.
        alpha: See function detnet.
        batch_size: Number of images used for one mini-batch.

    Returns:
        None. Automatically saves training log and the best model in
        directories "./logs" and "./models" respectively.
    """
    # Create output names
    today = datetime.date.today().strftime("%Y%m%d")
    bname_dataset = pink.io.extract_basename(dataset)
    bname_file = f"{today}_{alpha}"
    bname_output = os.path.join(output, "detnet", bname_dataset)

    for folder in ["models", "logs", "metrics"]:
        os.makedirs(os.path.join(bname_output, folder), exist_ok=True)

    fname_model = os.path.join(bname_output, "models", f"{bname_file}.h5")
    fname_logs = os.path.join(bname_output, "logs", f"{bname_file}.csv")
    fname_metrics = os.path.join(bname_output, "metrics", f"{bname_file}.csv")
    fname_plots = os.path.join(bname_output, "metrics", f"{bname_file}.pdf")

    # Verbose output
    print(f"Starting training of {bname_file}.")

    # Import data from prepared dataset
    x_train, y_train, x_valid, y_valid, _, _ = pink.io.load_npz(dataset)
    if x_train.dtype == float:
        x_train = skimage.util.img_as_ubyte(x_train)
    if x_valid.dtype == float:
        x_valid = skimage.util.img_as_ubyte(x_valid)

    # Convert Images
    # Normalization isn't mentioned in publication
    train_imgs = x_train.astype(np.float32)
    train_imgs /= 255
    valid_imgs = x_valid.astype(np.float32)
    valid_imgs /= 255

    # Convert masks
    train_masks = get_seg_maps(y_train, img_size)
    valid_masks = get_seg_maps(y_valid, img_size)

    # Parameters
    data_params = {
        "batch_size": batch_size,
        "image_size": img_size,
        "shuffle": True,
    }

    # Data Generators
    training_generator = DataGenerator(valid_masks, train_masks, **data_params)
    validation_generator = DataGenerator(valid_imgs, valid_masks, **data_params)

    # compilation
    model = detnet(image_size=img_size, alpha=alpha)
    model.compile(
        optimizer=pink.optimizers.amsgrad(learning_rate=0.001),
        loss=soft_dice,
        metrics=[f1_score, rmse_score],
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10),
        tf.keras.callbacks.ModelCheckpoint(fname_model, save_best_only=True),
        tf.keras.callbacks.CSVLogger(fname_logs),
    ]

    # Fitting
    model.fit(
        training_generator,
        validation_data=validation_generator,
        epochs=250,
        callbacks=callbacks,
        verbose=2,
    )

    # Verbose output
    print(f"Training for {bname_file} complete.")

    # Prediction
    pred_masks = [model.predict(m[None, :, :, None]) for m in valid_imgs]
    pred_coords = [get_coordinates(m) for m in pred_masks]

    # Metric calculation per image
    df = pd.DataFrame()
    for i, (pred, true) in enumerate(zip(pred_coords, y_valid)):
        curr_df = compute_metrics(pred, true)
        curr_df["image"] = i
        df = df.append(curr_df)
    df.to_csv(fname_metrics)

    plot_metrics(fname_plots, df)

    return df["f1_integral"].mean()


def main():
    # Argparse
    args = _parse_args()
    dataset = args.dataset
    output = args.output

    # Training and evaluation
    alphas = np.round(np.linspace(0, 1, 20), 4)
    best_hparam = {
        "score": 0,
        "alpha": 0,
    }

    for alpha in alphas:
        f1_score = run_train_eval(
            dataset=dataset, output=output, img_size=512, alpha=alpha, batch_size=8
        )
        if f1_score > best_hparam["score"]:
            best_hparam["score"] = f1_score
            best_hparam["alpha"] = alpha

    print(
        textwrap.dedent(
            f"""Optimal metrics found are:
        * Score: {best_hparam["score"]}
        * Alpha: {best_hparam["alpha"]}"""
        )
    )


if __name__ == "__main__":
    main()

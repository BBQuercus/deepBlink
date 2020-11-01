import argparse
import glob
import itertools
import os
import shutil

import deepblink as pink
import numpy as np
import pandas as pd
import skimage.measure
import skimage.morphology
import skimage.util
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import wandb


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--savedir")
    args = parser.parse_args()
    return args


def get_coordinates(mask: np.ndarray) -> np.ndarray:
    """Segmentation mask -> coordinate list."""
    binary = np.round(mask.squeeze())
    label = skimage.measure.label(binary)
    props = skimage.measure.regionprops(label)
    coords = np.array([p.centroid for p in props])
    return coords


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
        conv = tf.keras.layers.Conv2D(filters, **conv_args)(inputs)
        norm = tfa.layers.InstanceNormalization()(conv)
        return norm

    conv_args = {"kernel_size": (3, 3), "padding": "same", "activation": "relu"}

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


class DetNet:
    def __init__(
        self,
        savedir: str,
        bname_file: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        img_size: int,
        alpha: float,
        epochs: int,
    ):
        self.savedir = savedir
        self.bname_file = bname_file
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.img_size = img_size
        self.alpha = alpha
        self.epochs = epochs

    def __call__(self):
        self.touch()
        self.create()
        self.train()
        self.evaluate()
        return self.score

    def touch(self):
        self.fname_model = os.path.join(self.savedir, "models", f"{self.bname_file}.h5")
        self.fname_logs = os.path.join(self.savedir, "logs", f"{self.bname_file}.csv")
        self.fname_metrics = os.path.join(
            self.savedir, "metrics", f"{self.bname_file}.csv"
        )

    def create(self):
        # Convert Images
        # Assumed as normalization isn't mentioned in publication
        train_imgs = self.x_train.astype(np.float32)
        train_imgs /= 255
        valid_imgs = self.x_valid.astype(np.float32)
        valid_imgs /= 255

        # Convert coordinates to masks
        train_masks = get_seg_maps(self.y_train, self.img_size)
        valid_masks = get_seg_maps(self.y_valid, self.img_size)

        # Parameters
        data_params = {
            "batch_size": 8,
            "image_size": self.img_size,
            "shuffle": True,
        }

        # Data Generators
        training_generator = DataGenerator(valid_masks, train_masks, **data_params)
        validation_generator = DataGenerator(valid_imgs, valid_masks, **data_params)

        self.valid_imgs = valid_imgs
        self.training_generator = training_generator
        self.validation_generator = validation_generator

    def train(self):
        # compilation
        model = detnet(image_size=self.img_size, alpha=self.alpha)
        model.compile(
            optimizer=pink.optimizers.amsgrad(learning_rate=0.001),
            loss=soft_dice,
            metrics=[f1_score, rmse_score],
        )

        wandb.init(name=self.bname_file, project="detnet")

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10),
            tf.keras.callbacks.ModelCheckpoint(self.fname_model, save_best_only=True),
            tf.keras.callbacks.CSVLogger(self.fname_logs),
            wandb.keras.WandbCallback(),
        ]

        # Fitting
        model.fit(
            self.training_generator,
            validation_data=self.validation_generator,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=2,
        )

        self.model = model

    def evaluate(self):
        pred_masks = [self.model.predict(i[None, :, :, None]) for i in self.valid_imgs]
        pred_coords = [get_coordinates(m) for m in pred_masks]

        # Metric calculation per image
        df = pd.DataFrame()
        for i, (pred, true) in enumerate(zip(pred_coords, self.y_valid)):
            curr_df = pink.metrics.compute_metrics(pred, true, mdist=3)
            curr_df["image"] = i
            df = df.append(curr_df)
        df.to_csv(self.fname_metrics)

        self.score = df["f1_integral"].mean()


def run_sweep(
    dataset: str, savedir: str, alphas: iter = None,
):
    print(f"Running detnet on {dataset}.")

    # Import data
    x_train, y_train, x_valid, y_valid, _, _ = pink.io.load_npz(dataset)
    if x_train.dtype == float:
        x_train = skimage.util.img_as_ubyte(x_train)
    if x_valid.dtype == float:
        x_valid = skimage.util.img_as_ubyte(x_valid)

    # Create output names
    for folder in ["models", "logs", "metrics", "best"]:
        os.makedirs(os.path.join(savedir, folder), exist_ok=True)

    # Setup the scoreboard
    params = {
        "fname": None,
        "score": 0,
        "alpha": None,
    }

    # Sweep on detnet
    for alpha in alphas:
        bname_file = f"detnet_{pink.io.basename(dataset)}_{alpha}"
        dn = DetNet(
            savedir,
            bname_file,
            x_train,
            y_train,
            x_valid,
            y_valid,
            img_size=512,
            alpha=alpha,
            epochs=200,
        )
        score = dn()

        if score > params["score"]:
            params["fname"] = bname_file
            params["score"] = score
            params["alpha"] = alpha

    # Save the best model
    with open(os.path.join(savedir, "best", "info.txt"), "a") as f:
        f.write(f"For datasets {dataset}...\n")
        f.write(f"...best model: {params['fname']}\n")
        f.write(f"...best score: {params['score']}\n")

    shutil.copy(
        os.path.join(savedir, "models", f"{params['fname']}.h5"),
        os.path.join(savedir, "best"),
    )


if __name__ == "__main__":
    args = _parse_args()
    alphas = np.round(np.linspace(0, 1, 20), 4)

    run_sweep(args.dataset, args.savedir, alphas=alphas)

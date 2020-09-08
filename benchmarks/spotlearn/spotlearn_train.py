"""Train spotlearn on given dataset with automatic hyperparameter search."""
import datetime
import itertools
import os
import sys
import textwrap

import deepblink as pink
import numpy as np
import pandas as pd
import skimage.measure
import skimage.morphology
import skimage.util
import tensorflow as tf
import tensorflow.keras.backend as K

sys.path.append("../")
from util import _parse_args
from util import compute_metrics
from util import get_coordinates
from util import plot_metrics

K.set_image_data_format("channels_last")


def get_segmentation_map(
    coordinate_list: np.ndarray, img_size: int, obj_size: int, obj_type: str
) -> np.ndarray:
    """Convert coordinate list to segmentation map."""
    seg_map = np.ones((img_size, img_size))

    if obj_type == "diamond":
        obj = 1 - skimage.morphology.diamond(obj_size)
    elif obj_type == "disk":
        obj = 1 - skimage.morphology.disk(obj_size)
    else:
        raise ValueError(f"Object type must be valid. {obj_type} is not.")

    for r, c in np.round(coordinate_list).astype(int):
        seg_map[
            max(r - obj_size, 0) : r + obj_size + 1,
            max(c - obj_size, 0) : c + obj_size + 1,
        ] *= obj[
            max(obj_size - r, 0) : obj_size + img_size - r,
            max(obj_size - c, 0) : obj_size + img_size - c,
        ]
    return 1 - seg_map


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet_short_dropout(img_size=512):
    inputs = tf.keras.Input((img_size, img_size, 1))
    conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(
        inputs
    )
    conv1 = tf.keras.layers.Dropout(0.2)(conv1)
    conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(
        pool1
    )
    conv2 = tf.keras.layers.Dropout(0.2)(conv2)
    conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(
        conv2
    )
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(
        pool2
    )
    conv3 = tf.keras.layers.Dropout(0.2)(conv3)
    conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(
        conv3
    )

    up3 = tf.keras.layers.concatenate(
        [
            tf.keras.layers.Conv2DTranspose(
                128, (2, 2), strides=(2, 2), padding="same"
            )(conv3),
            conv2,
        ],
        axis=3,
    )
    conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(up3)
    conv4 = tf.keras.layers.Dropout(0.2)(conv4)
    conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(
        conv4
    )

    up4 = tf.keras.layers.concatenate(
        [
            tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(
                conv4
            ),
            conv1,
        ],
        axis=3,
    )
    conv5 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(up4)
    conv5 = tf.keras.layers.Dropout(0.2)(conv5)
    conv5 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(conv5)

    conv6 = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(conv5)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[conv6])

    return model


def get_callbacks(fname_model, fname_logs):
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        fname_model, monitor="val_loss", save_best_only=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=50, min_lr=0.001, verbose=1
    )
    model_es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.00000001, patience=10, verbose=1, mode="auto"
    )
    csv_logger = tf.keras.callbacks.CSVLogger(fname_logs, append=True)
    callbacks = [model_checkpoint, reduce_lr, model_es, csv_logger]
    return callbacks


def run_train_eval(
    dataset: str, output: str, img_size: int, obj_size: int, obj_type: str
) -> None:
    # Create output names
    today = datetime.date.today().strftime("%Y%m%d")
    bname_dataset = pink.io.extract_basename(dataset)
    bname_file = f"{today}_{obj_type}_{obj_size}"
    bname_output = os.path.join(output, "spotlearn", bname_dataset)

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

    # Convert images
    train_imgs = x_train.astype(np.float32)
    train_imgs /= 255
    train_imgs = np.expand_dims(train_imgs, 3)
    valid_imgs = x_valid.astype(np.float32)
    valid_imgs /= 255
    valid_imgs = np.expand_dims(valid_imgs, 3)

    # Convert masks
    train_masks = np.array(
        [get_segmentation_map(y, img_size, obj_size, obj_type) for y in y_train],
        dtype=np.float32,
    )
    train_masks = np.expand_dims(train_masks, 3)
    valid_masks = np.array(
        [get_segmentation_map(y, img_size, obj_size, obj_type) for y in y_valid],
        dtype=np.float32,
    )
    valid_masks = np.expand_dims(valid_masks, 3)

    # Compilation
    model = get_unet_short_dropout(img_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-5),
        loss=dice_coef_loss,
        metrics=[dice_coef],
    )

    # Callbacks
    callbacks = get_callbacks(fname_model, fname_logs)

    # Fitting
    model.fit(
        train_imgs,
        train_masks,
        batch_size=8,
        epochs=250,
        verbose=2,
        shuffle=True,
        validation_data=(valid_imgs, valid_masks),
        callbacks=callbacks,
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
    sizes = [0, 1, 2, 3]
    shapes = ["diamond", "disk"]
    best_hparam = {
        "score": 0,
        "size": 0,
        "shape": 0,
    }

    for size, shape in itertools.product(sizes, shapes):
        f1_score = run_train_eval(
            dataset=dataset, output=output, img_size=512, obj_size=size, obj_type=shape
        )
        if f1_score > best_hparam["score"]:
            best_hparam["score"] = f1_score
            best_hparam["size"] = size
            best_hparam["shape"] = shape

    print(
        textwrap.dedent(
            f"""Optimal metrics found are:
        * Score: {best_hparam["score"]}
        * Size: {best_hparam["size"]}
        * Shape: {best_hparam["shape"]}"""
        )
    )


if __name__ == "__main__":
    main()

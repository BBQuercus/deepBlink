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
import wandb

K.set_image_data_format("channels_last")


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
    conv_args = {"kernel_size": (3, 3), "activation": "relu", "padding": "same"}
    convt_args = {"kernel_size": (2, 2), "strides": (2, 2), "padding": "same"}

    inputs = tf.keras.Input((img_size, img_size, 1))
    conv1 = tf.keras.layers.Conv2D(64, **conv_args)(inputs)
    conv1 = tf.keras.layers.Dropout(0.2)(conv1)
    conv1 = tf.keras.layers.Conv2D(64, **conv_args)(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(128, **conv_args)(pool1)
    conv2 = tf.keras.layers.Dropout(0.2)(conv2)
    conv2 = tf.keras.layers.Conv2D(128, **conv_args)(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(256, **conv_args)(pool2)
    conv3 = tf.keras.layers.Dropout(0.2)(conv3)
    conv3 = tf.keras.layers.Conv2D(256, **conv_args)(conv3)

    up3 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv2DTranspose(128, **convt_args)(conv3), conv2], axis=3,
    )
    conv4 = tf.keras.layers.Conv2D(128, **conv_args)(up3)
    conv4 = tf.keras.layers.Dropout(0.2)(conv4)
    conv4 = tf.keras.layers.Conv2D(128, **conv_args)(conv4)

    up4 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv2DTranspose(64, **convt_args)(conv4), conv1], axis=3,
    )
    conv5 = tf.keras.layers.Conv2D(64, **conv_args)(up4)
    conv5 = tf.keras.layers.Dropout(0.2)(conv5)
    conv5 = tf.keras.layers.Conv2D(64, **conv_args)(conv5)

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


class SpotLearn:
    def __init__(
        self,
        savedir: str,
        bname_file: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        img_size: int,
        obj_size: int,
        obj_type: str,
        epochs: int,
    ):
        self.savedir = savedir
        self.bname_file = bname_file
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.img_size = img_size
        self.obj_size = obj_size
        self.obj_type = obj_type
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
        # Convert images
        train_imgs = self.x_train.astype(np.float32)
        train_imgs /= 255
        train_imgs = np.expand_dims(train_imgs, 3)
        valid_imgs = self.x_valid.astype(np.float32)
        valid_imgs /= 255
        valid_imgs = np.expand_dims(valid_imgs, 3)

        # Convert masks
        train_masks = np.array(
            [
                get_segmentation_map(y, self.img_size, self.obj_size, self.obj_type)
                for y in self.y_train
            ],
            dtype=np.float32,
        )
        train_masks = np.expand_dims(train_masks, 3)
        valid_masks = np.array(
            [
                get_segmentation_map(y, self.img_size, self.obj_size, self.obj_type)
                for y in self.y_valid
            ],
            dtype=np.float32,
        )
        valid_masks = np.expand_dims(valid_masks, 3)

        self.train_imgs = train_imgs
        self.train_masks = train_masks
        self.valid_imgs = valid_imgs
        self.valid_masks = valid_masks

    def train(self):
        # Compilation
        model = get_unet_short_dropout(self.img_size)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-5),
            loss=dice_coef_loss,
            metrics=[dice_coef],
        )

        wandb.init(project="spotlearn", name=self.bname_file)

        # Callbacks
        callbacks = get_callbacks(self.fname_model, self.fname_logs)
        callbacks.append(wandb.keras.WandbCallback())

        # Fitting
        model.fit(
            self.train_imgs,
            self.train_masks,
            batch_size=2,
            epochs=self.epochs,
            verbose=2,
            shuffle=True,
            validation_data=(self.valid_imgs, self.valid_masks),
            callbacks=callbacks,
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


def run_sweep(dataset: str, savedir: str, sizes: int, shapes: int):
    print(f"Running spotlearn on {dataset}.")

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
        "size": None,
        "type": None,
    }

    # Sweep on spotlearn
    for size, shape in itertools.product(sizes, shapes):
        bname_file = f"spotlearn_{pink.io.basename(dataset)}_{shape}_{size}"
        sl = SpotLearn(
            savedir,
            bname_file,
            x_train,
            y_train,
            x_valid,
            y_valid,
            img_size=512,
            obj_size=size,
            obj_type=shape,
            epochs=200,
        )
        score = sl()

        if score > params["score"]:
            params["fname"] = bname_file
            params["score"] = score
            params["size"] = size
            params["type"] = shape

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
    sizes = [0, 1, 2, 3]
    shapes = ["diamond", "disk"]

    run_sweep(args.dataset, args.savedir, sizes=sizes, shapes=shapes)

"""Generate YAML config files for all performance experiments.

Each config patches exactly one (or a few) settings relative to the baseline,
keeping experiments fully reproducible from their YAML alone.
"""

import argparse
import copy
import os

import yaml


BASELINE = {
    "name": "deepBlink-perf",
    "run_name": "baseline",
    "savedir": "experiments/models",
    "use_wandb": False,
    "augmentation_args": {
        "flip": False,
        "illuminate": False,
        "rotate": False,
        "gaussian_noise": False,
        "translate": False,
    },
    "dataset": "SpotsDataset",
    "dataset_args": {
        "name": None,  # filled from --dataset
        "cell_size": 4,
        "smooth_factor": 1,
    },
    "model": "SpotsModel",
    "network": "unet",
    "network_args": {
        "dropout": 0.3,
        "filters": 5,
        "ndown": 2,
        "l2": 1e-6,
        "block": "convolutional",
    },
    "loss": "combined_dice_rmse",
    "optimizer": "amsgrad",
    "train_args": {
        "batch_size": 2,
        "epochs": 200,
        "learning_rate": 1e-4,
        "overfit": False,
        "pre_train": None,
    },
}


def _make_experiment(exp_id, name, patches):
    """Create a config dict by patching the baseline."""
    cfg = copy.deepcopy(BASELINE)
    cfg["run_name"] = name

    for key_path, value in patches.items():
        keys = key_path.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value

    return exp_id, name, cfg


EXPERIMENTS = [
    (0, "baseline", {}),
    (1, "aug_all", {
        "augmentation_args.flip": True,
        "augmentation_args.illuminate": True,
        "augmentation_args.rotate": True,
        "augmentation_args.gaussian_noise": True,
        "augmentation_args.translate": True,
    }),
    (2, "aug_flip_rotate", {
        "augmentation_args.flip": True,
        "augmentation_args.rotate": True,
    }),
    (3, "lr_cosine", {
        "train_args.learning_rate": 1e-3,
        "train_args.lr_schedule": "cosine",
        "train_args.lr_min": 1e-6,
    }),
    (4, "lr_plateau", {
        "train_args.lr_schedule": "plateau",
    }),
    (5, "lr_warmup_cosine", {
        "train_args.learning_rate": 1e-3,
        "train_args.lr_schedule": "warmup_cosine",
        "train_args.warmup_epochs": 10,
        "train_args.lr_min": 1e-6,
    }),
    (6, "bs8", {
        "train_args.batch_size": 8,
        "train_args.learning_rate": 4e-4,
    }),
    (7, "bs16", {
        "train_args.batch_size": 16,
        "train_args.learning_rate": 8e-4,
    }),
    (8, "early_stop", {
        "train_args.early_stopping": True,
        "train_args.early_stopping_patience": 20,
    }),
    (9, "focal_loss", {
        "loss": "combined_focal_rmse",
    }),
    (10, "smooth_09", {
        "dataset_args.smooth_factor": 0.9,
    }),
    (11, "combined_best", {
        "augmentation_args.flip": True,
        "augmentation_args.illuminate": True,
        "augmentation_args.rotate": True,
        "augmentation_args.gaussian_noise": True,
        "augmentation_args.translate": True,
        "train_args.learning_rate": 4e-4,
        "train_args.lr_schedule": "warmup_cosine",
        "train_args.warmup_epochs": 10,
        "train_args.lr_min": 1e-6,
        "train_args.batch_size": 8,
        "train_args.early_stopping": True,
        "train_args.early_stopping_patience": 20,
        "loss": "combined_focal_rmse",
        "dataset_args.smooth_factor": 0.9,
    }),
    (12, "mixed_precision", {
        "train_args.mixed_precision": True,
    }),
    (13, "tta", {}),  # TTA is eval-only, training is baseline
    (14, "combined_best_tta", {
        "augmentation_args.flip": True,
        "augmentation_args.illuminate": True,
        "augmentation_args.rotate": True,
        "augmentation_args.gaussian_noise": True,
        "augmentation_args.translate": True,
        "train_args.learning_rate": 4e-4,
        "train_args.lr_schedule": "warmup_cosine",
        "train_args.warmup_epochs": 10,
        "train_args.lr_min": 1e-6,
        "train_args.batch_size": 8,
        "train_args.early_stopping": True,
        "train_args.early_stopping_patience": 20,
        "loss": "combined_focal_rmse",
        "dataset_args.smooth_factor": 0.9,
    }),
]


def main():
    parser = argparse.ArgumentParser(description="Generate experiment configs")
    parser.add_argument("--dataset", required=True, help="Path to dataset .npz file")
    parser.add_argument(
        "--outdir", default="experiments/configs", help="Output directory"
    )
    parser.add_argument(
        "--experiment", type=int, default=None,
        help="Generate config for a single experiment ID only",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    experiments = EXPERIMENTS
    if args.experiment is not None:
        experiments = [e for e in experiments if e[0] == args.experiment]
        if not experiments:
            raise ValueError(f"Experiment ID {args.experiment} not found")

    for exp_id, name, patches in experiments:
        _, _, cfg = _make_experiment(exp_id, name, patches)
        cfg["dataset_args"]["name"] = os.path.abspath(args.dataset)

        fname = os.path.join(args.outdir, f"exp_{exp_id:02d}_{name}.yaml")
        with open(fname, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"  wrote {fname}")


if __name__ == "__main__":
    main()

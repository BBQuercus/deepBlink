"""Train all 9 deepBlink models on GPU.

Usage:
    python train_all_models.py              # train all 9
    python train_all_models.py vesicle      # train just vesicle
"""
import os
import sys
import time
import shutil

# Set CUDA library paths before importing TF
NVIDIA_LIB_DIR = "./.venv/lib/python3.12/site-packages/nvidia"
ld_paths = [
    f"{NVIDIA_LIB_DIR}/{pkg}/lib"
    for pkg in ["cudnn", "cuda_runtime", "cublas", "cufft", "cusolver", "cusparse", "nvjitlink"]
]
os.environ["LD_LIBRARY_PATH"] = ":".join(ld_paths) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import tensorflow as tf

# Enable memory growth to avoid OOM
gpu_devices = tf.config.list_physical_devices("GPU")
if gpu_devices:
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    print(f"GPU: {gpu_devices[0].name}")
else:
    print("WARNING: No GPU found, training on CPU")

import deepblink

DATASET_DIR = "./datasets"
OUTPUT_DIR = "./retrained_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 9 models -> 6 datasets
MODEL_DATASET_MAP = {
    "vesicle":     "vesicle.npz",
    "receptor":    "receptor.npz",
    "microtubule": "microtubule.npz",
    "particle":    "particle.npz",
    "smfish1":     "smfish.npz",
    "suntag1":     "suntag.npz",
    "suntag2":     "suntag.npz",
    "smfish2":     "smfish.npz",
    "ms2":         "smfish.npz",
}

BASE_CONFIG = {
    "name": "deepBlink",
    "use_wandb": False,
    "savedir": OUTPUT_DIR,
    "augmentation_args": {
        "flip": True,
        "illuminate": True,
        "rotate": True,
        "gaussian_noise": True,
        "translate": True,
    },
    "dataset": "SpotsDataset",
    "dataset_args": {"cell_size": 4, "smooth_factor": 1},
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
        "batch_size": 4,
        "epochs": 200,
        "learning_rate": 1e-4,
        "overfit": False,
        "pre_train": None,
    },
}


def train_one(model_name: str) -> str | None:
    """Train a single model. Returns output path or None on failure."""
    dataset_file = MODEL_DATASET_MAP[model_name]
    dataset_path = os.path.join(DATASET_DIR, dataset_file)
    final_path = os.path.join(OUTPUT_DIR, f"{model_name}.h5")

    if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
        print(f"SKIP {model_name}: {final_path} already exists")
        return final_path

    cfg = {**BASE_CONFIG}
    cfg["run_name"] = model_name
    cfg["dataset_args"] = {**BASE_CONFIG["dataset_args"], "name": dataset_path}

    print(f"\n{'='*60}")
    print(f" TRAINING: {model_name}  (dataset: {dataset_file})")
    print(f"{'='*60}\n")

    t0 = time.time()
    deepblink.train(cfg)
    elapsed = time.time() - t0
    print(f"\n  {model_name} trained in {elapsed/60:.1f} min")

    # Rename timestamped file -> clean name
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(f"_{model_name}.h5"):
            src = os.path.join(OUTPUT_DIR, f)
            shutil.move(src, final_path)
            print(f"  Renamed {f} -> {model_name}.h5")
            break

    return final_path if os.path.exists(final_path) else None


if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else list(MODEL_DATASET_MAP.keys())

    results = {}
    total_t0 = time.time()
    for name in models:
        if name not in MODEL_DATASET_MAP:
            print(f"Unknown model: {name}")
            continue
        try:
            path = train_one(name)
            results[name] = "OK" if path else "FAILED"
        except Exception as e:
            print(f"ERROR training {name}: {e}")
            import traceback; traceback.print_exc()
            results[name] = f"ERROR: {e}"

    total_elapsed = time.time() - total_t0
    print(f"\n\n{'='*60}")
    print(f" SUMMARY  (total: {total_elapsed/60:.1f} min)")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"  {name:15s} {status}")

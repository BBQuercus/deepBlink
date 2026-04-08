"""Thin wrapper: load a YAML config and run a training experiment."""

import argparse
import os
import sys
import time

import yaml

# Ensure deepblink is importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deepblink.training import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Train a single experiment")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument(
        "--savedir", default=None,
        help="Override savedir from config",
    )
    parser.add_argument(
        "--run-name", default=None,
        help="Override run_name from config",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.savedir:
        cfg["savedir"] = args.savedir
    if args.run_name:
        cfg["run_name"] = args.run_name

    os.makedirs(cfg["savedir"], exist_ok=True)

    print(f"=== Experiment: {cfg['run_name']} ===")
    print(f"Config: {args.config}")
    print(f"Dataset: {cfg['dataset_args']['name']}")
    print(f"Loss: {cfg['loss']}")
    print(f"LR: {cfg['train_args']['learning_rate']}")
    print(f"Batch size: {cfg['train_args']['batch_size']}")
    print(f"Epochs: {cfg['train_args']['epochs']}")
    print()

    t0 = time.perf_counter()
    run_experiment(cfg)
    elapsed = time.perf_counter() - t0

    print(f"\n=== Done in {elapsed / 60:.1f} min ===")


if __name__ == "__main__":
    main()

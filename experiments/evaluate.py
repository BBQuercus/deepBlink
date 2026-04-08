"""Evaluate a trained model on the test split and write per-image metrics to CSV."""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deepblink.inference import predict, predict_tta
from deepblink.io import load_model, load_npz
from deepblink.metrics import compute_metrics


def evaluate_model(model_path, dataset_path, output_path, use_tta=False, mdist=3):
    """Run evaluation on the test split of a dataset."""
    model = load_model(model_path)
    x_test, y_test = load_npz(dataset_path, test_only=True)

    rows = []
    for i in range(len(x_test)):
        image = x_test[i].squeeze()

        # Ground truth coordinates
        true_coords = y_test[i]
        if true_coords.ndim > 1 and true_coords.shape[-1] >= 2:
            true_rc = true_coords[:, :2] if true_coords.ndim == 2 else true_coords
        else:
            true_rc = true_coords

        # Predict
        t0 = time.perf_counter()
        if use_tta:
            pred_coords = predict_tta(image, model, probability=0.5)
        else:
            pred_coords = predict(image, model, probability=None)
        infer_ms = (time.perf_counter() - t0) * 1000

        # Ensure 2D coordinates only
        if pred_coords.ndim == 2 and pred_coords.shape[1] > 2:
            pred_rc = pred_coords[:, :2]
        else:
            pred_rc = pred_coords

        # Compute metrics
        if len(pred_rc) == 0 or len(true_rc) == 0:
            rows.append({
                "image_idx": i,
                "f1_3px": 0.0,
                "f1_integral": 0.0,
                "mean_euclidean": float("nan"),
                "inference_ms": infer_ms,
                "n_pred": len(pred_rc),
                "n_true": len(true_rc),
            })
            continue

        df_metrics = compute_metrics(pred_rc, true_rc, mdist=mdist)

        # F1 at max cutoff (3px)
        f1_at_3 = df_metrics["f1_score"].iloc[-1]
        f1_integral = df_metrics["f1_integral"].iloc[0]
        mean_euc = df_metrics["mean_euclidean"].iloc[0]

        rows.append({
            "image_idx": i,
            "f1_3px": f1_at_3,
            "f1_integral": f1_integral,
            "mean_euclidean": mean_euc,
            "inference_ms": infer_ms,
            "n_pred": len(pred_rc),
            "n_true": len(true_rc),
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} results to {output_path}")

    # Print summary
    print(f"\n  F1@3px:       {df['f1_3px'].mean():.4f} ± {df['f1_3px'].std():.4f}")
    print(f"  F1 integral:  {df['f1_integral'].mean():.4f}")
    print(f"  Mean euclid:  {df['mean_euclidean'].mean():.4f}")
    print(f"  Inference:    {df['inference_ms'].mean():.1f} ms/image")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model", required=True, help="Path to .h5 model")
    parser.add_argument("--dataset", required=True, help="Path to dataset .npz")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--tta", action="store_true", help="Use test-time augmentation")
    parser.add_argument("--mdist", type=float, default=3, help="Max distance for F1")
    args = parser.parse_args()

    evaluate_model(args.model, args.dataset, args.output, use_tta=args.tta, mdist=args.mdist)


if __name__ == "__main__":
    main()

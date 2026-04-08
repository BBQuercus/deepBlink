"""Collect per-experiment CSVs into a single summary table."""

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment results")
    parser.add_argument(
        "--results-dir", default="experiments/results", help="Directory with CSVs"
    )
    parser.add_argument(
        "--output", default="experiments/summary.csv", help="Output summary CSV"
    )
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.results_dir, "exp_*.csv")))
    if not csv_files:
        print(f"No result CSVs found in {args.results_dir}")
        return

    rows = []
    for csv_path in csv_files:
        name = os.path.splitext(os.path.basename(csv_path))[0]
        df = pd.read_csv(csv_path)
        rows.append({
            "experiment": name,
            "f1_3px_mean": df["f1_3px"].mean(),
            "f1_3px_std": df["f1_3px"].std(),
            "f1_integral_mean": df["f1_integral"].mean(),
            "mean_euclidean": df["mean_euclidean"].mean(),
            "infer_ms_per_img": df["inference_ms"].mean(),
            "n_images": len(df),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output, index=False)

    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(summary.to_string(index=False))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

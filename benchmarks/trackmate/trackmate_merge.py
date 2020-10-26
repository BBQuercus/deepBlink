import glob
import os

import pandas as pd

basedir = "/tungstenfs/scratch/shared/gchao_ggiorget/benchmarks/trackmate/"
folders = sorted(glob.glob(os.path.join(basedir, "*")))

results_trackmate = pd.DataFrame()
for folder in folders:
    curr_df = pd.concat(
        [
            pd.read_csv(f, index_col=0)
            for f in glob.glob(os.path.join(folder, "test_processed", "*"))
        ]
    )
    curr_df["name"] = os.path.basename(folder)
    results_trackmate = results_trackmate.append(curr_df)
results_trackmate.to_csv("trackmate.csv")

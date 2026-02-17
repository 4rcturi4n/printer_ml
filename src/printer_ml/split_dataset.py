import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_regression(
    in_csv: str,
    out_train: str,
    out_val: str,
    out_split_json: str,
    seed: int = 42,
    val_size: float = 0.2,
    n_bins: int = 4,
):
    df = pd.read_csv(in_csv)

    # Ensure video_id exists
    if "video_id" not in df.columns:
        if "video_name" not in df.columns:
            raise ValueError("Input CSV must contain video_id or video_name.")
        df["video_id"] = df["video_name"]

    # Clean target
    df["axial_resolution"] = pd.to_numeric(df["axial_resolution"], errors="coerce")
    df = df[(df["axial_resolution"].notna()) & (df["axial_resolution"] != 0)].copy()

    # Per-video target for stratification (mean axial per video -> then log)
    per_video = df.groupby("video_id", as_index=False)["axial_resolution"].mean()
    per_video["log_axial"] = np.log(per_video["axial_resolution"])

    # Bins ONLY for stratification (log distribution)
    bins = pd.qcut(per_video["log_axial"], q=n_bins, duplicates="drop")

    train_ids, val_ids = train_test_split(
        per_video["video_id"],
        test_size=val_size,
        random_state=seed,
        stratify=bins,
    )

    train_df = df[df["video_id"].isin(train_ids)].copy()
    val_df   = df[df["video_id"].isin(val_ids)].copy()

    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    os.makedirs(os.path.dirname(out_val), exist_ok=True)
    os.makedirs(os.path.dirname(out_split_json), exist_ok=True)

    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)

    split_meta = {
        "seed": seed,
        "val_size": val_size,
        "n_bins_requested": n_bins,
        "n_bins_actual": int(len(bins.cat.categories)),
        "train_video_ids": train_ids.tolist(),
        "val_video_ids": val_ids.tolist(),
    }
    with open(out_split_json, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)

    print("✅ Saved:", out_train, "| rows:", len(train_df), "| videos:", train_df["video_id"].nunique())
    print("✅ Saved:", out_val,   "| rows:", len(val_df),   "| videos:", val_df["video_id"].nunique())
    print("✅ Saved split ids:", out_split_json)

    # Quick check: bin counts in train/val (videos)
    train_bins = bins[per_video["video_id"].isin(train_ids)].value_counts().sort_index()
    val_bins   = bins[per_video["video_id"].isin(val_ids)].value_counts().sort_index()
    print("\nTrain bin counts (videos):\n", train_bins)
    print("\nVal bin counts (videos):\n", val_bins)

    return train_df, val_df, split_meta

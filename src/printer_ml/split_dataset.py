import os
import json
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
        raise ValueError("Input CSV must contain video_id")

    # Create bins from existing log column
    bins = pd.qcut(df["log_axial_resolution"], q=n_bins, duplicates="drop")

    # Split
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=bins,
    )

    # Bin counts after split
    train_bins = bins.loc[train_df.index].value_counts().sort_index()
    val_bins = bins.loc[val_df.index].value_counts().sort_index()

    # Create folders
    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    os.makedirs(os.path.dirname(out_val), exist_ok=True)
    os.makedirs(os.path.dirname(out_split_json), exist_ok=True)

    # Save CSVs
    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)

    split_meta = {
        "seed": seed,
        "val_size": val_size,
        "n_bins_requested": n_bins,
        "n_bins_actual": int(len(bins.cat.categories)),
        "train_indices": train_df.index.tolist(),
        "val_indices": val_df.index.tolist(),
        "train_bin_counts": {str(k): int(v) for k, v in train_bins.items()},
        "val_bin_counts": {str(k): int(v) for k, v in val_bins.items()},
    }

    with open(out_split_json, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)

    print("✅ Saved:", out_train, "| rows:", len(train_df))
    print("✅ Saved:", out_val, "| rows:", len(val_df))
    print("✅ Saved split ids:", out_split_json)

    print("\nTrain bin counts:\n", train_bins)
    print("\nVal bin counts:\n", val_bins)

    return train_df, val_df, split_meta

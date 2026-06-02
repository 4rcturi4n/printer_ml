import os
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def make_stratified_kfold_splits(
    in_csv: str,
    out_dir: str,
    n_splits: int = 5,
    n_bins: int = 4,
    seed: int = 42,
):
    df = pd.read_csv(in_csv)

    if "video_id" not in df.columns:
        raise ValueError("Input CSV must contain video_id")
    if "log_axial_resolution" not in df.columns:
        raise ValueError("Input CSV must contain log_axial_resolution")

    # assumes one row per video
    if len(df) != df["video_id"].nunique():
        raise ValueError(
            "This k-fold splitter assumes one row per video, but duplicate video_id values were found."
        )

    bins = pd.qcut(df["log_axial_resolution"], q=n_bins, duplicates="drop")
    y_strat = bins.cat.codes

    counts = pd.Series(y_strat).value_counts()
    if counts.min() < n_splits:
        raise ValueError(
            f"Cannot do StratifiedKFold with n_splits={n_splits}. "
            f"Smallest bin has only {int(counts.min())} samples. "
            f"Reduce n_bins or n_splits."
        )

    os.makedirs(out_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_infos = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y_strat), start=1):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        train_csv = os.path.join(out_dir, f"fold_{fold}_train.csv")
        val_csv = os.path.join(out_dir, f"fold_{fold}_val.csv")
        split_json = os.path.join(out_dir, f"fold_{fold}_split.json")

        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)

        train_bins = bins.iloc[train_idx].value_counts().sort_index()
        val_bins = bins.iloc[val_idx].value_counts().sort_index()

        meta = {
            "fold": fold,
            "seed": seed,
            "n_splits": n_splits,
            "n_bins_requested": n_bins,
            "n_bins_actual": int(len(bins.cat.categories)),
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "train_video_ids": train_df["video_id"].tolist(),
            "val_video_ids": val_df["video_id"].tolist(),
            "train_bin_counts": {str(k): int(v) for k, v in train_bins.items()},
            "val_bin_counts": {str(k): int(v) for k, v in val_bins.items()},
        }

        with open(split_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"✅ Fold {fold}")
        print(f"   train: {train_csv} | rows: {len(train_df)}")
        print(f"   val:   {val_csv} | rows: {len(val_df)}")
        print(f"   meta:  {split_json}")

        fold_infos.append({
            "fold": fold,
            "train_csv": train_csv,
            "val_csv": val_csv,
            "split_json": split_json,
        })

    return fold_infos
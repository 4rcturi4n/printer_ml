import os
import sys
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import torch


# --------------------------------------------------
# Make src import work
# --------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC = os.path.join(ROOT, "src")

sys.path.insert(0, SRC)
sys.path.insert(0, ROOT)


from configs.project import CROP_BOX

from printer_ml.kfold_X3DL import (
    set_seed,
    load_or_extract_x3d_l_embeddings,
    train_x3d_l_embedding_fold,
)


# --------------------------------------------------
# Config
# --------------------------------------------------

CFG = {
    # saved fold data
    "split_dir": os.path.join(ROOT, "data", "processed", "kfold_splits"),

    "video_col": "video_path",
    "target_col": "axial_resolution",

    # crop
    "crop_box": CROP_BOX,

    # output
    "out_dir": os.path.join(ROOT, "results", "x3d_l_saved_folds_64f"),

    # saved embeddings
    "embedding_dir": os.path.join(ROOT, "results", "x3d_l_saved_folds", "embeddings"),
    "overwrite_embeddings": False,

    # k-fold
    "n_splits": 5,

    # X3D-L frozen embedding extraction
    "model_name": "x3d_l",
    "num_frames": 64,
    "image_size": 312,
    "eps_sec": 1.0 / 30.0,
    "embedding_batch_size": 1,

    # regression head
    # Same style as DINOv2 baseline:
    # Linear -> ReLU -> Dropout -> Linear
    "hidden_dim": 256,
    "dropout": 0.2,

    # training regression head only
    "batch_size": 4,
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,

    # early stopping
    # Same logic as DINOv2:
    # monitor val_mae_phys
    "early_stopping_patience": 8,
    "min_delta": 0.0,

    # target transform
    # Same as DINOv2:
    # axial_resolution -> log -> normalize using train fold only
    "use_log_target": True,

    # dataloader workers for video embedding extraction
    "num_workers": 4,

    # reproducibility
    "seed": 42,
}


# --------------------------------------------------
# Find saved fold CSVs
# --------------------------------------------------

def get_fold_paths(split_dir, fold):
    """
    Same behavior as DINOv2 baseline.

    This script only loads saved folds.
    It does NOT create new folds.
    """

    candidates = [
        (
            os.path.join(split_dir, f"fold_{fold}_train.csv"),
            os.path.join(split_dir, f"fold_{fold}_val.csv"),
        ),
        (
            os.path.join(split_dir, f"train_fold_{fold}.csv"),
            os.path.join(split_dir, f"val_fold_{fold}.csv"),
        ),
        (
            os.path.join(split_dir, f"fold{fold}_train.csv"),
            os.path.join(split_dir, f"fold{fold}_val.csv"),
        ),
        (
            os.path.join(split_dir, f"train_{fold}.csv"),
            os.path.join(split_dir, f"val_{fold}.csv"),
        ),
    ]

    for train_csv, val_csv in candidates:
        if os.path.exists(train_csv) and os.path.exists(val_csv):
            return train_csv, val_csv

    raise FileNotFoundError(
        f"Could not find train/val CSVs for fold {fold} in {split_dir}.\n"
        f"Expected names like:\n"
        f"  fold_{fold}_train.csv\n"
        f"  fold_{fold}_val.csv\n"
    )


# --------------------------------------------------
# Main saved-fold training
# --------------------------------------------------

def train_x3d_l_saved_folds():
    cfg = deepcopy(CFG)

    set_seed(cfg["seed"])

    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(cfg["embedding_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Using split dir: {cfg['split_dir']}")
    print(f"Using crop box: {cfg['crop_box']}")
    print(f"Using model: {cfg['model_name']}")
    print("Backbone: frozen X3D-L")
    print("Training: regression head only")
    print("Best checkpoint metric: val_mae_phys")

    all_histories = []
    fold_summaries = []

    for fold in range(1, cfg["n_splits"] + 1):
        train_csv, val_csv = get_fold_paths(
            split_dir=cfg["split_dir"],
            fold=fold,
        )

        print()
        print("=" * 80)
        print(f"Preparing fold {fold}")
        print("=" * 80)
        print(f"Train CSV: {train_csv}")
        print(f"Val CSV:   {val_csv}")

        fold_embedding_dir = os.path.join(
            cfg["embedding_dir"],
            f"fold_{fold}",
        )
        os.makedirs(fold_embedding_dir, exist_ok=True)

        train_embedding_path = os.path.join(
            fold_embedding_dir,
            "train_embeddings.pt",
        )

        val_embedding_path = os.path.join(
            fold_embedding_dir,
            "val_embeddings.pt",
        )

        # --------------------------------------------------
        # Extract or reuse frozen X3D-L embeddings
        # --------------------------------------------------

        load_or_extract_x3d_l_embeddings(
            csv_path=train_csv,
            cfg=cfg,
            save_path=train_embedding_path,
            device=device,
        )

        load_or_extract_x3d_l_embeddings(
            csv_path=val_csv,
            cfg=cfg,
            save_path=val_embedding_path,
            device=device,
        )

        # --------------------------------------------------
        # Train regression head from saved embeddings
        # --------------------------------------------------

        history_df, fold_summary = train_x3d_l_embedding_fold(
            fold=fold,
            train_embedding_path=train_embedding_path,
            val_embedding_path=val_embedding_path,
            train_csv=train_csv,
            val_csv=val_csv,
            cfg=deepcopy(cfg),
            device=device,
        )

        all_histories.append(history_df)
        fold_summaries.append(fold_summary)

    # --------------------------------------------------
    # Save combined history
    # --------------------------------------------------

    all_history_df = pd.concat(
        all_histories,
        axis=0,
        ignore_index=True,
    )

    all_history_path = os.path.join(
        cfg["out_dir"],
        "history_all_folds.csv",
    )

    all_history_df.to_csv(all_history_path, index=False)

    # --------------------------------------------------
    # Save fold summary CSV
    # --------------------------------------------------

    summary_df = pd.DataFrame(fold_summaries)

    summary_csv_path = os.path.join(
        cfg["out_dir"],
        "x3d_l_saved_folds_summary.csv",
    )

    summary_df.to_csv(summary_csv_path, index=False)

    mae_values = summary_df["best_val_mae_phys"].values
    rmse_values = summary_df["best_val_rmse_phys"].values
    bias_values = summary_df["best_bias_phys"].values

    final_summary = {
        "model_name": cfg["model_name"],
        "n_splits": cfg["n_splits"],
        "mean_val_mae_phys": float(np.mean(mae_values)),
        "std_val_mae_phys": float(np.std(mae_values)),
        "mean_val_rmse_phys": float(np.mean(rmse_values)),
        "std_val_rmse_phys": float(np.std(rmse_values)),
        "mean_bias_phys": float(np.mean(bias_values)),
        "std_bias_phys": float(np.std(bias_values)),
        "folds": fold_summaries,
        "history_all_folds_csv": all_history_path,
        "kfold_summary_csv": summary_csv_path,
        "split_dir": cfg["split_dir"],
        "embedding_dir": cfg["embedding_dir"],
        "crop_box": cfg["crop_box"],
        "num_frames": cfg["num_frames"],
        "image_size": cfg["image_size"],
        "use_log_target": cfg["use_log_target"],
        "early_stopping_patience": cfg["early_stopping_patience"],
        "min_delta": cfg["min_delta"],
        "backbone_frozen": True,
        "best_checkpoint_metric": "val_mae_phys",
    }

    final_summary_path = os.path.join(
        cfg["out_dir"],
        "kfold_summary.json",
    )

    with open(final_summary_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    # --------------------------------------------------
    # Save config
    # --------------------------------------------------

    config_path = os.path.join(cfg["out_dir"], "config.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # --------------------------------------------------
    # Final print
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("X3D-L TRAINING ON SAVED FOLDS FINISHED")
    print("=" * 80)

    print(
        summary_df[
            [
                "fold",
                "best_epoch",
                "best_val_mae_phys",
                "best_val_rmse_phys",
                "best_bias_phys",
                "stopped_early",
                "stop_epoch",
            ]
        ]
    )

    print()
    print(f"Mean validation physical MAE:  {np.mean(mae_values):.5f}")
    print(f"Std validation physical MAE:   {np.std(mae_values):.5f}")
    print(f"Mean validation physical RMSE: {np.mean(rmse_values):.5f}")
    print(f"Std validation physical RMSE:  {np.std(rmse_values):.5f}")
    print()
    print(f"Saved all-fold history to: {all_history_path}")
    print(f"Saved k-fold summary to:   {summary_csv_path}")
    print(f"Saved final summary to:    {final_summary_path}")


if __name__ == "__main__":
    train_x3d_l_saved_folds()
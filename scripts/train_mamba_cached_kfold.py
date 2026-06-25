"""
Train the Mamba sequence head on cached, augmented DINOv2 embeddings.

Run scripts/extract_dinov2_embeddings_kfold.py FIRST to produce the
cached embeddings this script reads.

What this produces
-------------------
results/mamba_cached_kfold/
  config.json
  kfold_summary.json
  mamba_cached_summary.csv
  fold_1/
    model_best.pth
    history.csv
    summary.json
    mae_curve.png
    ept_distribution.png
    monitoring_plots/
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC = os.path.join(ROOT, "src")

sys.path.insert(0, SRC)
sys.path.insert(0, ROOT)

from printer_ml.mamba_early_prediction import set_seed
from printer_ml.mamba_cached_regression import train_cached_mamba_fold


CFG = {
    "embeddings_dir": os.path.join(ROOT, "results", "dinov2_embeddings"),
    "out_dir": os.path.join(ROOT, "results", "mamba_cached_kfold"),
    "n_splits": 5,

    # Mamba
    "n_mamba_layers": 2,
    "d_state": 16,
    "d_conv": 4,
    "expand": 2,

    "hidden_dim": 256,
    "dropout": 0.2,

    "batch_size": 4,
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "early_stopping_patience": 10,
    "min_delta": 0.0,

    "ept_threshold_um": 0.2,

    # step 4 — oversampling bins
    "n_bins": 4,

    # step 5 — embedding-space augmentation knobs
    "augment_kwargs": {
        "noise_prob": 0.8,
        "noise_sigma": 0.02,
        "mask_prob": 0.3,
        "mask_frac": 0.1,
        "warp_prob": 0.2,
        "warp_strength": 0.3,
        "mixup_prob": 0.2,
        "mixup_alpha": 0.3,
        "mixup_target_tol": 0.5,
    },

    "max_trajectory_plots": 10,
    "seed": 42,
}


def main():
    set_seed(CFG["seed"])
    os.makedirs(CFG["out_dir"], exist_ok=True)

    with open(os.path.join(CFG["out_dir"], "config.json"), "w", encoding="utf-8") as f:
        json.dump(CFG, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Mamba cached k-fold training (augmented + oversampled)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Embeddings dir: {CFG['embeddings_dir']}")
    print(f"Out dir: {CFG['out_dir']}")
    print()

    all_histories = []
    fold_summaries = []

    for fold in range(1, CFG["n_splits"] + 1):
        history_df, fold_summary = train_cached_mamba_fold(fold=fold, cfg=CFG, device=device)
        all_histories.append(history_df)
        fold_summaries.append(fold_summary)

    pd.concat(all_histories, ignore_index=True).to_csv(
        os.path.join(CFG["out_dir"], "history_all_folds.csv"), index=False
    )

    summary_df = pd.DataFrame(fold_summaries)
    summary_df.to_csv(os.path.join(CFG["out_dir"], "mamba_cached_summary.csv"), index=False)

    mae_values = summary_df["best_val_mae_phys"].values
    final_summary = {
        "n_splits": CFG["n_splits"],
        "mean_val_mae_phys": float(np.mean(mae_values)),
        "std_val_mae_phys": float(np.std(mae_values)),
        "folds": fold_summaries,
    }
    with open(os.path.join(CFG["out_dir"], "kfold_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    print()
    print("=" * 80)
    print("MAMBA CACHED K-FOLD TRAINING — FINISHED")
    print("=" * 80)
    print(summary_df[["fold", "best_epoch", "best_val_mae_phys"]].to_string(index=False))
    print(f"Mean final MAE: {np.mean(mae_values):.5f} ± {np.std(mae_values):.5f} μm")


if __name__ == "__main__":
    main()

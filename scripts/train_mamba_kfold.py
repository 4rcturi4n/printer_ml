"""
Train DINOv2 + Mamba early predictor with k-fold cross-validation.

What this produces
------------------
results/mamba_early_prediction/
  config.json
  kfold_summary.json
  history_all_folds.csv
  mamba_early_prediction_summary.csv
  mae_per_timestep_all_folds.csv       <- MAE at each frame %, averaged across folds
  mae_per_timestep_plot.png            <- cross-fold MAE curve

  fold_1/
    model_best.pth
    history.csv
    summary.json
    mae_per_timestep.csv               <- MAE/RMSE at each frame position
    ept_summary.csv                    <- EPT per video
    val_predictions_per_timestep.csv   <- full prediction trajectory per video
    mae_curve.png
    ept_distribution.png
    monitoring_plots/
      {video_id}_trajectory.png        <- prediction vs. time for each val video

Key metrics
-----------
- final_mae_phys     : MAE at the last frame (comparable to other models)
- mean_ept_pct       : average % of video needed before prediction stabilises
- pct_videos_ept_found: % of videos where the model detects the outcome early enough
"""

import os
import sys
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC  = os.path.join(ROOT, "src")

sys.path.insert(0, SRC)
sys.path.insert(0, ROOT)

from configs.project import CROP_BOX
from printer_ml.mamba_early_prediction import set_seed, train_mamba_fold


# --------------------------------------------------
# Config
# --------------------------------------------------

CFG = {
    # data
    "split_dir":  os.path.join(ROOT, "data", "processed", "kfold_splits"),
    "video_col":  "video_path",
    "target_col": "axial_resolution",
    "crop_box":   CROP_BOX,

    # output
    "out_dir":  os.path.join(ROOT, "results", "mamba_early_prediction"),
    "n_splits": 1,

    # DINOv2 (frozen spatial encoder)
    "dinov2_model": "dinov2_vits14",   # vits14 / vitb14 / vitl14
    "num_frames":   32,                # frames sampled per video
    "image_size":   224,

    # Mamba
    "n_mamba_layers": 2,
    "d_state":        16,
    "d_conv":          4,
    "expand":          2,

    # Regression head
    "hidden_dim": 256,
    "dropout":    0.2,

    # Training
    "batch_size":             1,
    "epochs":                100,
    "lr":               1e-3,
    "weight_decay":     1e-4,
    "early_stopping_patience": 10,
    "min_delta":        0.0,

    # Target transform
    "use_log_target": True,

    # Monitoring
    # Alert if |predicted - true| stays below this for the rest of the video.
    # Set to a physically meaningful deviation for your experiment (μm).
    "ept_threshold_um": 0.2,

    # Misc
    "num_workers":          4,   # increase on server if I/O is a bottleneck
    "seed":                42,
    "max_trajectory_plots": 10,   # per-video plots saved per fold
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def get_fold_paths(split_dir: str, fold: int):
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
    ]
    for train_csv, val_csv in candidates:
        if os.path.exists(train_csv) and os.path.exists(val_csv):
            return train_csv, val_csv
    raise FileNotFoundError(
        f"Could not find train/val CSVs for fold {fold} in {split_dir}.\n"
        f"Expected names like fold_{fold}_train.csv / fold_{fold}_val.csv"
    )


def plot_crossfold_mae(mae_dfs: list, out_path: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    for df in mae_dfs:
        fold_num = int(df["fold"].iloc[0])
        ax.plot(df["frame_pct"], df["mae_phys"], alpha=0.4, label=f"Fold {fold_num}")

    all_mae = pd.concat(mae_dfs, ignore_index=True)
    avg     = all_mae.groupby("frame_idx")[["frame_pct", "mae_phys"]].mean().reset_index()
    ax.plot(avg["frame_pct"], avg["mae_phys"],
            color="black", linewidth=2.5, label="Mean across folds")

    ax.set_xlabel("Video progress (%)")
    ax.set_ylabel("MAE (μm)")
    ax.set_title("MAE vs. Video Progress — All Folds\n(how early the model predicts accurately)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    set_seed(CFG["seed"])
    os.makedirs(CFG["out_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    print("=" * 80)
    print("Mamba Early Predictor — SANITY CHECK (single video overfit)")
    print("=" * 80)
    print(f"Device         : {device}")
    print(f"Out dir        : {CFG['out_dir']}")
    print()

    train_csv, _ = get_fold_paths(CFG["split_dir"], fold=1)
    single_video_df = pd.read_csv(train_csv).iloc[:1].reset_index(drop=True)

    print(f"Sanity check video: {single_video_df['video_path'].iloc[0]}")
    print(f"Target value      : {single_video_df['axial_resolution'].iloc[0]}")
    print()

    history_df, fold_summary = train_mamba_fold(
        fold=1,
        train_df=single_video_df,
        val_df=single_video_df,
        cfg={
            **CFG,
            "epochs": 100,
            "early_stopping_patience": 100,
            "batch_size": 1,
        },
        device=device,
    )

    print()
    print("=" * 80)
    print("SANITY CHECK RESULTS")
    print("=" * 80)
    print(history_df[["epoch", "train_loss", "final_mae_phys"]].to_string(index=False))
    print()
    print(f"Final train_loss : {history_df['train_loss'].iloc[-1]:.6f}")
    print(f"Best MAE (phys)  : {fold_summary['best_val_mae_phys']:.6f} μm")
    print()
    if fold_summary["best_val_mae_phys"] < 0.01:
        print("PASSED — model can overfit a single sample.")
    else:
        print("FAILED — model cannot overfit a single sample. Check your pipeline.")


if __name__ == "__main__":
    main()

"""
def main():
    set_seed(CFG["seed"])
    os.makedirs(CFG["out_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    print("=" * 80)
    print("Mamba Early Predictor — k-fold training")
    print("=" * 80)
    print(f"Device         : {device}")
    print(f"Split dir      : {CFG['split_dir']}")
    print(f"Out dir        : {CFG['out_dir']}")
    print(f"DINOv2 model   : {CFG['dinov2_model']}")
    print(f"Frames per video: {CFG['num_frames']}")
    print(f"Mamba layers   : {CFG['n_mamba_layers']}")
    print(f"EPT threshold  : {CFG['ept_threshold_um']} μm")
    print()
    

    all_histories  = []
    fold_summaries = []

    for fold in range(1, CFG["n_splits"] + 1):
        train_csv, val_csv = get_fold_paths(CFG["split_dir"], fold)

        print(f"Loading fold {fold}")
        print(f"  train: {train_csv}")
        print(f"  val:   {val_csv}")

        
        train_df = pd.read_csv(train_csv).reset_index(drop=True)
        val_df   = pd.read_csv(val_csv).reset_index(drop=True)

        history_df, fold_summary = train_mamba_fold(
            fold=fold,
            train_df=train_df,
            val_df=val_df,
            cfg=deepcopy(CFG),
            device=device,
        )

        fold_summary["train_csv"] = train_csv
        fold_summary["val_csv"]   = val_csv

        all_histories.append(history_df)
        fold_summaries.append(fold_summary)

    # --------------------------------------------------
    # Combined outputs
    # --------------------------------------------------

    pd.concat(all_histories, ignore_index=True).to_csv(
        os.path.join(CFG["out_dir"], "history_all_folds.csv"), index=False
    )

    summary_df = pd.DataFrame(fold_summaries)
    summary_df.to_csv(
        os.path.join(CFG["out_dir"], "mamba_early_prediction_summary.csv"), index=False
    )

    # Average MAE-per-timestep across folds + cross-fold plot
    mae_dfs = []
    for fold in range(1, CFG["n_splits"] + 1):
        p = os.path.join(CFG["out_dir"], f"fold_{fold}", "mae_per_timestep.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["fold"] = fold
            mae_dfs.append(df)

    if mae_dfs:
        all_mae_df = pd.concat(mae_dfs, ignore_index=True)
        avg_mae_df = (
            all_mae_df
            .groupby("frame_idx")[["frame_pct", "mae_phys", "rmse_phys"]]
            .mean()
            .reset_index()
        )
        avg_mae_df.to_csv(
            os.path.join(CFG["out_dir"], "mae_per_timestep_all_folds.csv"), index=False
        )
        plot_crossfold_mae(mae_dfs, os.path.join(CFG["out_dir"], "mae_per_timestep_plot.png"))

    # Final k-fold summary JSON
    mae_values = summary_df["best_val_mae_phys"].values
    ept_values = summary_df["mean_ept_pct"].values

    final_summary = {
        "n_splits":            CFG["n_splits"],
        "mean_val_mae_phys":   float(np.mean(mae_values)),
        "std_val_mae_phys":    float(np.std(mae_values)),
        "mean_ept_pct":        float(np.mean(ept_values)),
        "std_ept_pct":         float(np.std(ept_values)),
        "ept_threshold_um":    CFG["ept_threshold_um"],
        "folds":               fold_summaries,
    }

    with open(os.path.join(CFG["out_dir"], "kfold_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    with open(os.path.join(CFG["out_dir"], "config.json"), "w", encoding="utf-8") as f:
        json.dump(CFG, f, indent=2)

    # --------------------------------------------------
    # Final print
    # --------------------------------------------------
    print()
    print("=" * 80)
    print("MAMBA EARLY PREDICTION — FINISHED")
    print("=" * 80)
    print(summary_df[[
        "fold", "best_epoch", "best_val_mae_phys",
        "mean_ept_pct", "pct_videos_ept_found", "stopped_early",
    ]].to_string(index=False))
    print()
    print(f"Mean final MAE : {np.mean(mae_values):.5f} ± {np.std(mae_values):.5f} μm")
    print(f"Mean EPT       : {np.mean(ept_values):.1f} ± {np.std(ept_values):.1f}% of video")
    print()
    print(f"Results saved to: {CFG['out_dir']}")


if __name__ == "__main__":
    main()
"""
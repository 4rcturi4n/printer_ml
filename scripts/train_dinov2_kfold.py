import os
import sys
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader


# --------------------------------------------------
# Make src import work
# --------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC = os.path.join(ROOT, "src")

sys.path.insert(0, SRC)
sys.path.insert(0, ROOT)


from configs.project import CROP_BOX

from printer_ml.dinov2_regression import (
    set_seed,
    VideoRegressionDataset,
    DINOv2VideoRegressor,
    evaluate_regression,
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
    "out_dir": os.path.join(ROOT, "results", "dinov2_saved_folds"),

    # k-fold
    "n_splits": 5,

    # DINOv2
    "dinov2_model": "dinov2_vits14",
    "num_frames": 16,
    "image_size": 224,

    # regression head
    "hidden_dim": 256,
    "dropout": 0.2,

    # training
    "batch_size": 4,
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,

    # early stopping
    "early_stopping_patience": 8,
    "min_delta": 0.0,

    # target transform
    "use_log_target": True,

    # dataloader
    "num_workers": 4,

    # reproducibility
    "seed": 42,
}


# --------------------------------------------------
# Find saved fold CSVs
# --------------------------------------------------

def get_fold_paths(split_dir, fold):
    """
    Supports common fold CSV naming styles.
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
# Train one fold
# --------------------------------------------------

def train_one_fold(
    fold,
    train_df,
    val_df,
    cfg,
    device,
):
    print()
    print("=" * 80)
    print(f"Fold {fold}")
    print("=" * 80)

    fold_out_dir = os.path.join(cfg["out_dir"], f"fold_{fold}")
    os.makedirs(fold_out_dir, exist_ok=True)

    # --------------------------------------------------
    # Target normalization from this fold's TRAIN ONLY
    # --------------------------------------------------

    y_train = train_df[cfg["target_col"]].astype(float).values

    if cfg["use_log_target"]:
        y_train = np.log(y_train)

    target_mean = float(y_train.mean())
    target_std = float(y_train.std() + 1e-8)

    print(f"Train videos: {len(train_df)}")
    print(f"Val videos:   {len(val_df)}")
    print(f"Target mean:  {target_mean:.6f}")
    print(f"Target std:   {target_std:.6f}")
    print(f"Crop box:     {cfg['crop_box']}")
    print(f"Early stopping patience: {cfg['early_stopping_patience']}")

    # --------------------------------------------------
    # Datasets
    # --------------------------------------------------

    train_dataset = VideoRegressionDataset(
        df=train_df,
        video_col=cfg["video_col"],
        target_col=cfg["target_col"],
        num_frames=cfg["num_frames"],
        image_size=cfg["image_size"],
        target_mean=target_mean,
        target_std=target_std,
        use_log_target=cfg["use_log_target"],
        crop_box=cfg["crop_box"],
    )

    val_dataset = VideoRegressionDataset(
        df=val_df,
        video_col=cfg["video_col"],
        target_col=cfg["target_col"],
        num_frames=cfg["num_frames"],
        image_size=cfg["image_size"],
        target_mean=target_mean,
        target_std=target_std,
        use_log_target=cfg["use_log_target"],
        crop_box=cfg["crop_box"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    # --------------------------------------------------
    # Fresh model for this fold
    # --------------------------------------------------

    model = DINOv2VideoRegressor(
        dinov2_name=cfg["dinov2_model"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.regressor.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    loss_fn = torch.nn.SmoothL1Loss()

    best_val_mae_phys = float("inf")
    best_epoch = None
    history = []

    epochs_without_improvement = 0
    stopped_early = False
    stop_epoch = None

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()

        train_losses = []

        for videos, targets in train_loader:
            videos = videos.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            preds = model(videos)

            loss = loss_fn(preds, targets)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        val_metrics = evaluate_regression(
            model=model,
            loader=val_loader,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
            use_log_target=cfg["use_log_target"],
        )

        current_val_mae = val_metrics["val_mae_phys"]

        improved = current_val_mae < (best_val_mae_phys - cfg["min_delta"])

        if improved:
            best_val_mae_phys = current_val_mae
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "fold": fold,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "cfg": cfg,
                "target_mean": target_mean,
                "target_std": target_std,
                "best_val_mae_phys": best_val_mae_phys,
            }

            torch.save(
                checkpoint,
                os.path.join(fold_out_dir, "model_best.pth"),
            )

        else:
            epochs_without_improvement += 1

        row = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "best_val_mae_phys_so_far": best_val_mae_phys,
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
        }

        history.append(row)

        print(
            f"Fold {fold} | "
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.5f} | "
            f"val_mae_phys={val_metrics['val_mae_phys']:.5f} | "
            f"val_rmse_phys={val_metrics['val_rmse_phys']:.5f} | "
            f"bias_phys={val_metrics['bias_phys']:.5f} | "
            f"best={best_val_mae_phys:.5f} | "
            f"no_improve={epochs_without_improvement}/{cfg['early_stopping_patience']}"
        )

        if epochs_without_improvement >= cfg["early_stopping_patience"]:
            stopped_early = True
            stop_epoch = epoch

            print()
            print(
                f"Early stopping fold {fold} at epoch {epoch}. "
                f"Best epoch: {best_epoch}, "
                f"best val_mae_phys: {best_val_mae_phys:.5f}"
            )

            break

    # --------------------------------------------------
    # Save fold history
    # --------------------------------------------------

    history_df = pd.DataFrame(history)

    history_path = os.path.join(fold_out_dir, "history.csv")
    history_df.to_csv(history_path, index=False)

    fold_summary = {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_mae_phys": float(best_val_mae_phys),
        "history_csv": history_path,
        "model_best": os.path.join(fold_out_dir, "model_best.pth"),
        "target_mean": target_mean,
        "target_std": target_std,
        "num_train": len(train_df),
        "num_val": len(val_df),
        "crop_box": cfg["crop_box"],
        "stopped_early": stopped_early,
        "stop_epoch": stop_epoch,
        "early_stopping_patience": cfg["early_stopping_patience"],
        "min_delta": cfg["min_delta"],
    }

    with open(os.path.join(fold_out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)

    return history_df, fold_summary


# --------------------------------------------------
# Main saved-fold training
# --------------------------------------------------

def train_saved_folds():
    set_seed(CFG["seed"])

    os.makedirs(CFG["out_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Using split dir: {CFG['split_dir']}")
    print(f"Using crop box: {CFG['crop_box']}")

    all_histories = []
    fold_summaries = []

    for fold in range(1, CFG["n_splits"] + 1):
        train_csv, val_csv = get_fold_paths(
            split_dir=CFG["split_dir"],
            fold=fold,
        )

        print()
        print(f"Loading fold {fold}")
        print(f"Train CSV: {train_csv}")
        print(f"Val CSV:   {val_csv}")

        train_df = pd.read_csv(train_csv).reset_index(drop=True)
        val_df = pd.read_csv(val_csv).reset_index(drop=True)

        history_df, fold_summary = train_one_fold(
            fold=fold,
            train_df=train_df,
            val_df=val_df,
            cfg=deepcopy(CFG),
            device=device,
        )

        fold_summary["train_csv"] = train_csv
        fold_summary["val_csv"] = val_csv

        all_histories.append(history_df)
        fold_summaries.append(fold_summary)

    # --------------------------------------------------
    # Save combined history
    # --------------------------------------------------

    all_history_df = pd.concat(all_histories, axis=0, ignore_index=True)

    all_history_path = os.path.join(CFG["out_dir"], "history_all_folds.csv")
    all_history_df.to_csv(all_history_path, index=False)

    # --------------------------------------------------
    # Save fold summary
    # --------------------------------------------------

    summary_df = pd.DataFrame(fold_summaries)

    summary_csv_path = os.path.join(
        CFG["out_dir"],
        "dinov2_saved_folds_summary.csv"
    )

    summary_df.to_csv(summary_csv_path, index=False)

    mae_values = summary_df["best_val_mae_phys"].values

    final_summary = {
        "n_splits": CFG["n_splits"],
        "mean_val_mae_phys": float(np.mean(mae_values)),
        "std_val_mae_phys": float(np.std(mae_values)),
        "folds": fold_summaries,
        "history_all_folds_csv": all_history_path,
        "kfold_summary_csv": summary_csv_path,
        "split_dir": CFG["split_dir"],
        "crop_box": CFG["crop_box"],
        "early_stopping_patience": CFG["early_stopping_patience"],
        "min_delta": CFG["min_delta"],
    }

    final_summary_path = os.path.join(CFG["out_dir"], "kfold_summary.json")

    with open(final_summary_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    # Save config
    config_path = os.path.join(CFG["out_dir"], "config.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(CFG, f, indent=2)

    print()
    print("=" * 80)
    print("DINOv2 TRAINING ON SAVED FOLDS FINISHED")
    print("=" * 80)
    print(summary_df[[
        "fold",
        "best_epoch",
        "best_val_mae_phys",
        "stopped_early",
        "stop_epoch",
    ]])
    print()
    print(f"Mean validation physical MAE: {np.mean(mae_values):.5f}")
    print(f"Std validation physical MAE:  {np.std(mae_values):.5f}")
    print()
    print(f"Saved all-fold history to: {all_history_path}")
    print(f"Saved k-fold summary to:   {summary_csv_path}")
    print(f"Saved final summary to:    {final_summary_path}")


if __name__ == "__main__":
    train_saved_folds()
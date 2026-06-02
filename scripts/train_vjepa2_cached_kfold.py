import os
import sys
import json
import time
from copy import deepcopy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# --------------------------------------------------
# Make src import work
# --------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC = os.path.join(ROOT, "src")

sys.path.insert(0, SRC)
sys.path.insert(0, ROOT)


from printer_ml.vjepa2_cached_regression import (
    set_seed,
    CachedEmbeddingDataset,
    EmbeddingRegressor,
    evaluate_cached_regression,
    predict_cached_physical,
)


# --------------------------------------------------
# Config
# --------------------------------------------------

CFG = {
    "embeddings_dir": os.path.join(ROOT, "results", "vjepa2_embeddings"),
    "out_dir": os.path.join(ROOT, "results", "vjepa2_cached_kfold"),

    "n_splits": 5,

    "hidden_dim": 256,
    "dropout": 0.2,

    "epochs": 50,
    "batch_size": 8,
    "lr": 1e-3,
    "weight_decay": 1e-4,

    "early_stopping_patience": 8,
    "min_delta": 0.0,

    "num_workers": 0,

    "seed": 42,
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):
    model.train()

    losses = []

    for embeddings, targets in loader:
        embeddings = embeddings.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        preds = model(embeddings)
        loss = criterion(preds, targets)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


def get_embedding_paths(embeddings_dir, fold):
    fold_dir = os.path.join(embeddings_dir, f"fold_{fold}")

    train_path = os.path.join(fold_dir, "train_embeddings.pt")
    val_path = os.path.join(fold_dir, "val_embeddings.pt")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train embeddings not found: {train_path}")

    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation embeddings not found: {val_path}")

    return train_path, val_path


# --------------------------------------------------
# Train one fold
# --------------------------------------------------

def train_one_fold(
    fold,
    cfg,
    device,
):
    print()
    print("=" * 80)
    print(f"Fold {fold}/{cfg['n_splits']}")
    print("=" * 80)

    fold_out_dir = os.path.join(cfg["out_dir"], f"fold_{fold}")
    os.makedirs(fold_out_dir, exist_ok=True)

    train_embeddings_path, val_embeddings_path = get_embedding_paths(
        embeddings_dir=cfg["embeddings_dir"],
        fold=fold,
    )

    train_dataset = CachedEmbeddingDataset(train_embeddings_path)
    val_dataset = CachedEmbeddingDataset(val_embeddings_path)

    target_mean = train_dataset.target_mean
    target_std = train_dataset.target_std
    use_log_target = train_dataset.use_log_target

    embed_dim = train_dataset.embeddings.shape[1]

    print(f"Train embeddings: {train_embeddings_path}")
    print(f"Val embeddings:   {val_embeddings_path}")
    print(f"Train samples:    {len(train_dataset)}")
    print(f"Val samples:      {len(val_dataset)}")
    print(f"Embedding dim:    {embed_dim}")
    print(f"Target mean:      {target_mean:.6f}")
    print(f"Target std:       {target_std:.6f}")

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

    model = EmbeddingRegressor(
        embed_dim=embed_dim,
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    criterion = nn.SmoothL1Loss()

    best_val_mae = float("inf")
    best_epoch = None
    best_state = None

    epochs_without_improvement = 0
    stopped_early = False
    stop_epoch = None

    history = []

    start_time = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_metrics = evaluate_cached_regression(
            model=model,
            loader=val_loader,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
            use_log_target=use_log_target,
        )

        current_val_mae = val_metrics["val_mae_phys"]

        improved = current_val_mae < (best_val_mae - cfg["min_delta"])

        if improved:
            best_val_mae = current_val_mae
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0

            torch.save(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "cfg": cfg,
                    "embed_dim": embed_dim,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "use_log_target": use_log_target,
                    "best_val_mae_phys": best_val_mae,
                    "train_embeddings_path": train_embeddings_path,
                    "val_embeddings_path": val_embeddings_path,
                },
                os.path.join(fold_out_dir, "model_best.pth"),
            )
        else:
            epochs_without_improvement += 1

        row = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "best_val_mae_phys_so_far": best_val_mae,
            "best_epoch": best_epoch,
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
            f"best={best_val_mae:.5f} | "
            f"no_improve={epochs_without_improvement}/{cfg['early_stopping_patience']}"
        )

        if epochs_without_improvement >= cfg["early_stopping_patience"]:
            stopped_early = True
            stop_epoch = epoch

            print()
            print(
                f"Early stopping fold {fold} at epoch {epoch}. "
                f"Best epoch: {best_epoch}, "
                f"best val_mae_phys: {best_val_mae:.5f}"
            )

            break

    elapsed = time.time() - start_time

    history_df = pd.DataFrame(history)
    history_csv = os.path.join(fold_out_dir, "history.csv")
    history_df.to_csv(history_csv, index=False)

    if best_state is not None:
        model.load_state_dict(best_state)

    preds_phys, targets_phys = predict_cached_physical(
        model=model,
        loader=val_loader,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        use_log_target=use_log_target,
    )

    pred_df = val_dataset.df.copy()
    pred_df["target_phys"] = targets_phys
    pred_df["pred_phys"] = preds_phys
    pred_df["error_phys"] = pred_df["pred_phys"] - pred_df["target_phys"]
    pred_df["abs_error_phys"] = pred_df["error_phys"].abs()

    pred_csv = os.path.join(fold_out_dir, "val_predictions.csv")
    pred_df.to_csv(pred_csv, index=False)

    best_row = history_df.loc[
        history_df["val_mae_phys"].idxmin()
    ].to_dict()

    fold_summary = {
        "fold": fold,
        "best_epoch": int(best_epoch),
        "best_val_mae_phys": float(best_val_mae),
        "best_val_rmse_phys": float(best_row["val_rmse_phys"]),
        "best_bias_phys": float(best_row["bias_phys"]),
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "use_log_target": bool(use_log_target),
        "embed_dim": int(embed_dim),
        "num_train": len(train_dataset),
        "num_val": len(val_dataset),
        "elapsed_seconds": float(elapsed),
        "history_csv": history_csv,
        "predictions_csv": pred_csv,
        "best_model_path": os.path.join(fold_out_dir, "model_best.pth"),
        "train_embeddings_path": train_embeddings_path,
        "val_embeddings_path": val_embeddings_path,
        "stopped_early": stopped_early,
        "stop_epoch": stop_epoch,
    }

    save_json(
        fold_summary,
        os.path.join(fold_out_dir, "summary.json"),
    )

    return history_df, fold_summary


# --------------------------------------------------
# Main
# --------------------------------------------------

def train_cached_kfold(cfg):
    set_seed(cfg["seed"])

    os.makedirs(cfg["out_dir"], exist_ok=True)

    save_json(cfg, os.path.join(cfg["out_dir"], "config.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("V-JEPA2 cached-embedding k-fold regression")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Embeddings dir: {cfg['embeddings_dir']}")
    print(f"Output dir: {cfg['out_dir']}")
    print("=" * 80)

    all_histories = []
    fold_summaries = []

    for fold in range(1, cfg["n_splits"] + 1):
        history_df, fold_summary = train_one_fold(
            fold=fold,
            cfg=deepcopy(cfg),
            device=device,
        )

        all_histories.append(history_df)
        fold_summaries.append(fold_summary)

    all_history_df = pd.concat(all_histories, axis=0, ignore_index=True)

    all_history_csv = os.path.join(cfg["out_dir"], "history_all_folds.csv")
    all_history_df.to_csv(all_history_csv, index=False)

    results_df = pd.DataFrame(fold_summaries)

    results_csv = os.path.join(cfg["out_dir"], "vjepa2_cached_kfold_results.csv")
    results_df.to_csv(results_csv, index=False)

    summary = {
        "experiment_name": "vjepa2_cached_kfold",
        "out_dir": cfg["out_dir"],
        "embeddings_dir": cfg["embeddings_dir"],
        "results_csv": results_csv,
        "history_all_folds_csv": all_history_csv,
        "n_splits": cfg["n_splits"],
        "metrics": {
            "val_mae_phys": {
                "mean": float(results_df["best_val_mae_phys"].mean()),
                "std": float(results_df["best_val_mae_phys"].std()),
            },
            "val_rmse_phys": {
                "mean": float(results_df["best_val_rmse_phys"].mean()),
                "std": float(results_df["best_val_rmse_phys"].std()),
            },
            "bias_phys": {
                "mean": float(results_df["best_bias_phys"].mean()),
                "std": float(results_df["best_bias_phys"].std()),
            },
        },
        "folds": fold_summaries,
    }

    summary_path = os.path.join(cfg["out_dir"], "summary.json")
    save_json(summary, summary_path)

    print()
    print("=" * 80)
    print("V-JEPA2 cached-embedding k-fold training finished")
    print("=" * 80)
    print(results_df[[
        "fold",
        "best_epoch",
        "best_val_mae_phys",
        "best_val_rmse_phys",
        "best_bias_phys",
        "stopped_early",
        "stop_epoch",
    ]])
    print()
    print(json.dumps(summary["metrics"], indent=2))
    print()
    print(f"Saved results to: {results_csv}")
    print(f"Saved summary to: {summary_path}")

    return summary


if __name__ == "__main__":
    train_cached_kfold(CFG)
"""
Train MambaSequenceRegressor on cached DINOv2 embeddings (no video I/O,
no backbone forward pass during training).

Combines:
  - step 3: variants already baked into the cached .pt file
            (see extract_dinov2_embeddings_kfold.py)
  - step 4: WeightedRandomSampler over target bins, so rare-target rows
            are drawn as often as common ones regardless of row count
  - step 5: EmbeddingAugmenter applied fresh every __getitem__ call
"""

import os
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from printer_ml.augmentation import EmbeddingAugmenter
from printer_ml.mamba_early_prediction import (
    MambaSequenceRegressor,
    ramp_weighted_loss,
    evaluate_early_prediction,
    plot_mae_curve,
    plot_ept_distribution,
    plot_video_trajectories,
)


# --------------------------------------------------
# Dataset over a cached fold_X/{train,val}_embeddings.pt file
# --------------------------------------------------

class CachedMambaEmbeddingDataset(Dataset):
    """
    Returns the same tuple shape as FrameSequenceDataset so the existing
    ramp_weighted_loss / evaluate_early_prediction functions work unchanged:
        embeddings   [T, D]
        target_norm  scalar
        target_phys  scalar
        video_id     str
    """

    def __init__(self, payload_path: str, augment: bool = False, augmenter: EmbeddingAugmenter = None):
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)

        self.embeddings = payload["embeddings"].float()       # [N, T, D]
        self.targets = payload["targets"].float()             # [N] normalized
        self.targets_phys = payload["targets_phys"].float()   # [N] physical
        self.video_ids = payload["video_ids"]

        self.target_mean = float(payload["target_mean"])
        self.target_std = float(payload["target_std"])
        self.use_log_target = bool(payload["use_log_target"])

        self.augment = augment
        self.augmenter = augmenter or EmbeddingAugmenter()

        # pool for mixup: (embedding, target_phys) pairs, built once
        self._mixup_pool = [
            (self.embeddings[i], float(self.targets_phys[i]))
            for i in range(len(self.embeddings))
        ]

    def __len__(self):
        return self.embeddings.shape[0]

    def bin_sample_weights(self, n_bins: int = 4) -> torch.Tensor:
        """
        Step 4: inverse-frequency weight per row, based on which target
        quantile bin it falls in. Feed this to WeightedRandomSampler.
        """
        y = np.log(self.targets_phys.numpy())
        bins = pd.qcut(y, q=n_bins, duplicates="drop")
        codes = bins.codes
        counts = np.bincount(codes)
        weight_per_bin = 1.0 / np.maximum(counts, 1)
        weights = weight_per_bin[codes]
        return torch.tensor(weights, dtype=torch.double)

    def __getitem__(self, idx):
        emb = self.embeddings[idx].clone()
        target_phys = float(self.targets_phys[idx])

        if self.augment:
            emb, target_phys = self.augmenter(
                emb, target_phys, mixup_pool=self._mixup_pool
            )

        y_log = np.log(target_phys) if self.use_log_target else target_phys
        target_norm = (y_log - self.target_mean) / self.target_std

        video_id = self.video_ids[idx]

        return (
            emb,
            torch.tensor(target_norm, dtype=torch.float32),
            torch.tensor(target_phys, dtype=torch.float32),
            str(video_id),
        )


# --------------------------------------------------
# Train one fold on cached embeddings
# --------------------------------------------------

def train_cached_mamba_fold(fold: int, cfg: dict, device: torch.device):
    fold_dir = os.path.join(cfg["embeddings_dir"], f"fold_{fold}")
    train_path = os.path.join(fold_dir, "train_embeddings.pt")
    val_path = os.path.join(fold_dir, "val_embeddings.pt")

    out_dir = os.path.join(cfg["out_dir"], f"fold_{fold}")
    os.makedirs(out_dir, exist_ok=True)

    augmenter = EmbeddingAugmenter(**cfg["augment_kwargs"])

    train_dataset = CachedMambaEmbeddingDataset(train_path, augment=True, augmenter=augmenter)
    val_dataset = CachedMambaEmbeddingDataset(val_path, augment=False)

    print(f"Fold {fold} | train rows (incl. variants): {len(train_dataset)} | val videos: {len(val_dataset)}")

    sample_weights = train_dataset.bin_sample_weights(n_bins=cfg["n_bins"])
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], sampler=sampler,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=0,
    )

    embed_dim = train_dataset.embeddings.shape[-1]
    model = MambaSequenceRegressor(
        embed_dim=embed_dim,
        n_mamba_layers=cfg["n_mamba_layers"],
        d_state=cfg["d_state"],
        d_conv=cfg["d_conv"],
        expand=cfg["expand"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.SmoothL1Loss()

    target_mean = train_dataset.target_mean
    target_std = train_dataset.target_std
    use_log_target = train_dataset.use_log_target

    best_val_mae = float("inf")
    best_epoch = None
    best_metrics = None
    history = []
    epochs_no_imp = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_losses = []

        for emb, targets_norm, targets_phys, video_ids in train_loader:
            emb = emb.to(device)
            targets_norm = targets_norm.to(device)

            optimizer.zero_grad()
            preds = model(emb)
            loss = ramp_weighted_loss(preds, targets_norm, loss_fn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        val_metrics, _, mae_per_t_df, ept_df = evaluate_early_prediction(
            model=model, loader=val_loader, device=device,
            target_mean=target_mean, target_std=target_std,
            use_log_target=use_log_target,
            ept_threshold_um=cfg["ept_threshold_um"],
        )

        current_mae = val_metrics["final_mae_phys"]
        improved = current_mae < (best_val_mae - cfg["min_delta"])

        if improved:
            best_val_mae = current_mae
            best_epoch = epoch
            best_metrics = deepcopy(val_metrics)
            epochs_no_imp = 0
            torch.save({
                "fold": fold, "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "cfg": cfg,
                "target_mean": target_mean, "target_std": target_std,
                "best_val_mae_phys": best_val_mae,
                "best_metrics": best_metrics,
            }, os.path.join(out_dir, "model_best.pth"))
            mae_per_t_df.to_csv(os.path.join(out_dir, "mae_per_timestep.csv"), index=False)
            ept_df.to_csv(os.path.join(out_dir, "ept_summary.csv"), index=False)
        else:
            epochs_no_imp += 1

        history.append({
            "fold": fold, "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "best_val_mae_phys_so_far": best_val_mae,
            "improved": improved,
            "epochs_without_improvement": epochs_no_imp,
        })

        print(
            f"Fold {fold} | Epoch {epoch:03d} | train_loss={train_loss:.5f} | "
            f"val_mae={current_mae:.5f} | best={best_val_mae:.5f} | "
            f"no_improve={epochs_no_imp}/{cfg['early_stopping_patience']}"
        )

        if epochs_no_imp >= cfg["early_stopping_patience"]:
            print(f"Early stopping fold {fold} at epoch {epoch}. Best epoch: {best_epoch}, MAE: {best_val_mae:.5f}")
            break

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)

    ckpt = torch.load(os.path.join(out_dir, "model_best.pth"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    best_val_metrics, per_video_df, mae_per_t_df, ept_df = evaluate_early_prediction(
        model=model, loader=val_loader, device=device,
        target_mean=target_mean, target_std=target_std,
        use_log_target=use_log_target,
        ept_threshold_um=cfg["ept_threshold_um"],
    )

    per_video_df.to_csv(os.path.join(out_dir, "val_predictions_per_timestep.csv"), index=False)
    plot_mae_curve(mae_per_t_df, os.path.join(out_dir, "mae_curve.png"), fold)
    plot_ept_distribution(ept_df, os.path.join(out_dir, "ept_distribution.png"), fold, cfg["ept_threshold_um"])
    plot_video_trajectories(
        per_video_df, out_dir=os.path.join(out_dir, "monitoring_plots"),
        ept_threshold_um=cfg["ept_threshold_um"], max_videos=cfg.get("max_trajectory_plots", 10),
    )

    fold_summary = {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_mae_phys": float(best_val_mae),
        "best_val_rmse_phys": float(best_val_metrics["final_rmse_phys"]),
        "num_train_rows": len(train_dataset),
        "num_val": len(val_dataset),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)

    return pd.DataFrame(history), fold_summary

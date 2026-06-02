import os
import json
import random
from copy import deepcopy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.models.hub import x3d_l

from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# X3D-L video dataset for embedding extraction
# --------------------------------------------------

class X3DLVideoEmbeddingDataset(Dataset):
    """
    Dataset used only for frozen X3D-L embedding extraction.

    It returns:
        video tensor: [C, T, H, W]
        physical target: axial_resolution
        video_path
    """

    def __init__(
        self,
        df: pd.DataFrame,
        video_col: str,
        target_col: str,
        crop_box,
        num_frames: int = 16,
        image_size: int = 312,
        eps_sec: float = 1.0 / 30.0,
    ):
        self.df = df.reset_index(drop=True)
        self.video_col = video_col
        self.target_col = target_col
        self.crop_box = crop_box
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.eps_sec = float(eps_sec)

        # X3D / PyTorchVideo normalization
        self.mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)

    def __len__(self):
        return len(self.df)

    def _uniform_timestamps(self, duration: float):
        """
        Deterministic uniform timestamps across the whole video.
        No random sampling, because embeddings must be reusable.
        """
        duration = float(duration)
        t0 = 0.0
        t1 = max(0.0, duration - 1e-3)

        if self.num_frames <= 1 or t1 <= t0:
            return [0.0]

        segment = (t1 - t0) / self.num_frames

        timestamps = []
        for i in range(self.num_frames):
            seg_start = t0 + i * segment
            seg_end = min(t1, seg_start + segment)
            t = 0.5 * (seg_start + seg_end)
            timestamps.append(float(t))

        return timestamps

    def _spatial_process(self, clip: torch.Tensor):
        """
        Input:
            clip: [C, T, H, W], uint8 or float

        Output:
            clip: [C, T, image_size, image_size], normalized float32
        """
        x1, y1, x2, y2 = self.crop_box

        clip = clip[:, :, y1:y2, x1:x2]

        # [C, T, H, W] -> [T, C, H, W]
        clip = clip.permute(1, 0, 2, 3)

        clip = F.interpolate(
            clip,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        # [T, C, H, W] -> [C, T, H, W]
        clip = clip.permute(1, 0, 2, 3)

        clip = clip.to(torch.float32) / 255.0
        clip = (clip - self.mean) / self.std

        return clip

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        video_path = row[self.video_col]
        target_phys = float(row[self.target_col])

        video = EncodedVideo.from_path(video_path)
        duration = float(video.duration)

        timestamps = self._uniform_timestamps(duration)

        frames = []
        for t in timestamps:
            start_sec = float(t)
            end_sec = float(min(duration, t + self.eps_sec))

            clip_dict = video.get_clip(
                start_sec=start_sec,
                end_sec=end_sec,
            )

            raw = clip_dict.get("video", None)

            if raw is None or raw.numel() == 0:
                continue

            # Take first frame from tiny timestamp clip.
            frames.append(raw[:, 0:1])  # [C, 1, H, W]

        if len(frames) == 0:
            # Fallback: small clip from the beginning.
            raw_clip = video.get_clip(
                start_sec=0.0,
                end_sec=min(duration, 2.0),
            )["video"]
        else:
            raw_clip = torch.cat(frames, dim=1)  # [C, T, H, W]

        C, T_raw, H, W = raw_clip.shape

        if T_raw >= self.num_frames:
            raw_clip = raw_clip[:, :self.num_frames]
        else:
            last_frame = raw_clip[:, -1:].repeat(
                1,
                self.num_frames - T_raw,
                1,
                1,
            )
            raw_clip = torch.cat([raw_clip, last_frame], dim=1)

        clip = self._spatial_process(raw_clip)

        return clip, torch.tensor(target_phys, dtype=torch.float32), video_path


# --------------------------------------------------
# Frozen X3D-L feature extractor
# --------------------------------------------------

class X3DLFeatureExtractor(nn.Module):
    """
    Frozen X3D-L backbone.

    The final classification projection is replaced by Identity,
    so the model outputs video embeddings instead of class logits.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        self.backbone = x3d_l(pretrained=pretrained)

        self.embedding_dim = self.backbone.blocks[-1].proj.in_features
        self.backbone.blocks[-1].proj = nn.Identity()

        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W]

        Returns:
            embeddings: [B, D]
        """
        feats = self.backbone(x)

        if feats.ndim > 2:
            feats = torch.flatten(feats, start_dim=1)

        return feats


def build_x3d_l_feature_extractor(device: torch.device):
    model = X3DLFeatureExtractor(pretrained=True)
    model = model.to(device)
    model.eval()
    return model


# --------------------------------------------------
# Embedding extraction and saving
# --------------------------------------------------

@torch.no_grad()
def extract_x3d_l_embeddings(
    df: pd.DataFrame,
    cfg: dict,
    save_path: str,
    device: torch.device,
):
    """
    Extract frozen X3D-L embeddings once and save them.

    Saved file contains:
        embeddings: Tensor [N, D]
        targets_phys: Tensor [N]
        video_paths: list[str]
        cfg_subset: dict
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dataset = X3DLVideoEmbeddingDataset(
        df=df,
        video_col=cfg["video_col"],
        target_col=cfg["target_col"],
        crop_box=cfg["crop_box"],
        num_frames=cfg["num_frames"],
        image_size=cfg["image_size"],
        eps_sec=cfg["eps_sec"],
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["embedding_batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    model = build_x3d_l_feature_extractor(device)

    all_embeddings = []
    all_targets = []
    all_paths = []

    for videos, targets, paths in loader:
        videos = videos.to(device, non_blocking=True)

        embeddings = model(videos)

        all_embeddings.append(embeddings.cpu())
        all_targets.append(targets.cpu())
        all_paths.extend(list(paths))

    embeddings = torch.cat(all_embeddings, dim=0)
    targets_phys = torch.cat(all_targets, dim=0)

    payload = {
        "embeddings": embeddings,
        "targets_phys": targets_phys,
        "video_paths": all_paths,
        "cfg_subset": {
            "model_name": cfg["model_name"],
            "video_col": cfg["video_col"],
            "target_col": cfg["target_col"],
            "num_frames": cfg["num_frames"],
            "image_size": cfg["image_size"],
            "crop_box": cfg["crop_box"],
            "eps_sec": cfg["eps_sec"],
        },
    }

    torch.save(payload, save_path)

    meta_csv = os.path.splitext(save_path)[0] + "_metadata.csv"
    pd.DataFrame(
        {
            "video_path": all_paths,
            "axial_resolution": targets_phys.numpy(),
        }
    ).to_csv(meta_csv, index=False)

    return save_path


def load_or_extract_x3d_l_embeddings(
    csv_path: str,
    cfg: dict,
    save_path: str,
    device: torch.device,
):
    """
    Reuse saved embeddings unless overwrite_embeddings=True.
    """

    if os.path.exists(save_path) and not cfg.get("overwrite_embeddings", False):
        print(f"✅ Using existing embeddings: {save_path}")
        return save_path

    print(f"Extracting embeddings from: {csv_path}")
    print(f"Saving embeddings to:      {save_path}")

    df = pd.read_csv(csv_path).reset_index(drop=True)

    return extract_x3d_l_embeddings(
        df=df,
        cfg=cfg,
        save_path=save_path,
        device=device,
    )


# --------------------------------------------------
# Dataset from saved embeddings
# --------------------------------------------------

class SavedEmbeddingRegressionDataset(Dataset):
    def __init__(
        self,
        embedding_path: str,
        target_mean: float,
        target_std: float,
        use_log_target: bool = True,
    ):
        payload = torch.load(embedding_path, map_location="cpu")

        self.embeddings = payload["embeddings"].float()
        self.targets_phys = payload["targets_phys"].float()
        self.video_paths = payload["video_paths"]

        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.use_log_target = bool(use_log_target)

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        target_phys = float(self.targets_phys[idx])

        y = target_phys

        if self.use_log_target:
            y = np.log(y)

        y_norm = (y - self.target_mean) / self.target_std

        return (
            embedding,
            torch.tensor(y_norm, dtype=torch.float32),
            torch.tensor(target_phys, dtype=torch.float32),
            self.video_paths[idx],
        )


# --------------------------------------------------
# Same regression head style as DINOv2 baseline
# --------------------------------------------------

class X3DEmbeddingRegressor(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings):
        pred = self.regressor(embeddings)
        return pred.squeeze(1)


# --------------------------------------------------
# Target inverse transform
# --------------------------------------------------

def inverse_transform_target(
    y_norm,
    target_mean: float,
    target_std: float,
    use_log_target: bool = True,
):
    y = y_norm * target_std + target_mean

    if use_log_target:
        y = np.exp(y)

    return y


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

@torch.no_grad()
def evaluate_embedding_regression(
    model,
    loader,
    device,
    loss_fn,
    target_mean: float,
    target_std: float,
    use_log_target: bool = True,
):
    model.eval()

    preds_norm = []
    targets_norm = []
    video_paths = []

    val_losses = []

    for embeddings, targets, targets_phys, paths in loader:
        embeddings = embeddings.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        preds = model(embeddings)

        loss = loss_fn(preds, targets)
        val_losses.append(loss.item())

        preds_norm.append(preds.cpu().numpy())
        targets_norm.append(targets.cpu().numpy())
        video_paths.extend(list(paths))

    preds_norm = np.concatenate(preds_norm)
    targets_norm = np.concatenate(targets_norm)

    preds_phys = inverse_transform_target(
        preds_norm,
        target_mean,
        target_std,
        use_log_target,
    )

    targets_phys = inverse_transform_target(
        targets_norm,
        target_mean,
        target_std,
        use_log_target,
    )

    mae_phys = mean_absolute_error(targets_phys, preds_phys)
    mse_phys = mean_squared_error(targets_phys, preds_phys)
    rmse_phys = np.sqrt(mse_phys)
    bias_phys = np.mean(preds_phys - targets_phys)

    pred_df = pd.DataFrame(
        {
            "video_path": video_paths,
            "true_phys": targets_phys,
            "pred_phys": preds_phys,
            "err_phys": preds_phys - targets_phys,
            "abs_err_phys": np.abs(preds_phys - targets_phys),
        }
    )

    metrics = {
        "val_loss": float(np.mean(val_losses)),
        "val_mae_phys": float(mae_phys),
        "val_rmse_phys": float(rmse_phys),
        "bias_phys": float(bias_phys),
    }

    return metrics, pred_df


# --------------------------------------------------
# Train one fold from saved X3D-L embeddings
# --------------------------------------------------

def train_x3d_l_embedding_fold(
    fold: int,
    train_embedding_path: str,
    val_embedding_path: str,
    train_csv: str,
    val_csv: str,
    cfg: dict,
    device: torch.device,
):
    print()
    print("=" * 80)
    print(f"X3D-L Fold {fold}")
    print("=" * 80)

    fold_out_dir = os.path.join(cfg["out_dir"], f"fold_{fold}")
    os.makedirs(fold_out_dir, exist_ok=True)

    # --------------------------------------------------
    # Target normalization from TRAIN fold only
    # Same methodology as DINOv2 baseline
    # --------------------------------------------------

    train_df = pd.read_csv(train_csv)

    y_train = train_df[cfg["target_col"]].astype(float).values

    if cfg["use_log_target"]:
        y_train = np.log(y_train)

    target_mean = float(y_train.mean())
    target_std = float(y_train.std() + 1e-8)

    print(f"Train CSV:    {train_csv}")
    print(f"Val CSV:      {val_csv}")
    print(f"Train emb:    {train_embedding_path}")
    print(f"Val emb:      {val_embedding_path}")
    print(f"Target mean:  {target_mean:.6f}")
    print(f"Target std:   {target_std:.6f}")

    # --------------------------------------------------
    # Datasets
    # --------------------------------------------------

    train_dataset = SavedEmbeddingRegressionDataset(
        embedding_path=train_embedding_path,
        target_mean=target_mean,
        target_std=target_std,
        use_log_target=cfg["use_log_target"],
    )

    val_dataset = SavedEmbeddingRegressionDataset(
        embedding_path=val_embedding_path,
        target_mean=target_mean,
        target_std=target_std,
        use_log_target=cfg["use_log_target"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    embedding_dim = train_dataset.embeddings.shape[1]

    model = X3DEmbeddingRegressor(
        embedding_dim=embedding_dim,
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
    best_metrics = None

    history = []

    epochs_without_improvement = 0
    stopped_early = False
    stop_epoch = None

    # --------------------------------------------------
    # Training loop
    # Best checkpoint uses val_mae_phys
    # Early stopping also uses val_mae_phys
    # --------------------------------------------------

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()

        train_losses = []

        for embeddings, targets, targets_phys, paths in train_loader:
            embeddings = embeddings.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            preds = model(embeddings)

            loss = loss_fn(preds, targets)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        val_metrics, val_pred_df = evaluate_embedding_regression(
            model=model,
            loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            target_mean=target_mean,
            target_std=target_std,
            use_log_target=cfg["use_log_target"],
        )

        current_val_mae = val_metrics["val_mae_phys"]

        improved = current_val_mae < (
            best_val_mae_phys - cfg["min_delta"]
        )

        if improved:
            best_val_mae_phys = current_val_mae
            best_epoch = epoch
            best_metrics = deepcopy(val_metrics)
            epochs_without_improvement = 0

            checkpoint = {
                "fold": fold,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "cfg": cfg,
                "embedding_dim": embedding_dim,
                "target_mean": target_mean,
                "target_std": target_std,
                "best_val_mae_phys": best_val_mae_phys,
                "best_metrics": best_metrics,
            }

            torch.save(
                checkpoint,
                os.path.join(fold_out_dir, "model_best.pth"),
            )

            val_pred_df.to_csv(
                os.path.join(fold_out_dir, "val_predictions_best.csv"),
                index=False,
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
            f"val_loss={val_metrics['val_loss']:.5f} | "
            f"val_mae_phys={val_metrics['val_mae_phys']:.5f} | "
            f"val_rmse_phys={val_metrics['val_rmse_phys']:.5f} | "
            f"bias_phys={val_metrics['bias_phys']:.5f} | "
            f"best={best_val_mae_phys:.5f} | "
            f"no_improve={epochs_without_improvement}/"
            f"{cfg['early_stopping_patience']}"
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
    # Save history and summary
    # --------------------------------------------------

    history_df = pd.DataFrame(history)

    history_path = os.path.join(fold_out_dir, "history.csv")
    history_df.to_csv(history_path, index=False)

    if best_metrics is None:
        best_metrics = {
            "val_loss": float("nan"),
            "val_mae_phys": float("nan"),
            "val_rmse_phys": float("nan"),
            "bias_phys": float("nan"),
        }

    fold_summary = {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_metrics["val_loss"]),
        "best_val_mae_phys": float(best_metrics["val_mae_phys"]),
        "best_val_rmse_phys": float(best_metrics["val_rmse_phys"]),
        "best_bias_phys": float(best_metrics["bias_phys"]),
        "history_csv": history_path,
        "model_best": os.path.join(fold_out_dir, "model_best.pth"),
        "val_predictions_best": os.path.join(
            fold_out_dir,
            "val_predictions_best.csv",
        ),
        "train_embedding_path": train_embedding_path,
        "val_embedding_path": val_embedding_path,
        "target_mean": target_mean,
        "target_std": target_std,
        "num_train": len(train_dataset),
        "num_val": len(val_dataset),
        "crop_box": cfg["crop_box"],
        "stopped_early": stopped_early,
        "stop_epoch": stop_epoch,
        "early_stopping_patience": cfg["early_stopping_patience"],
        "min_delta": cfg["min_delta"],
        "train_csv": train_csv,
        "val_csv": val_csv,
    }

    with open(
        os.path.join(fold_out_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(fold_summary, f, indent=2)

    return history_df, fold_summary
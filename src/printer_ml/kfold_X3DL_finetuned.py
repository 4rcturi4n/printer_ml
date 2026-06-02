import os
import json
import time
import random
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.models.hub import x3d_xs, x3d_s, x3d_m, x3d_l

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
# Dataset
# --------------------------------------------------

class PrinterVideoDatasetX3DFinetune(Dataset):
    """
    X3D video dataset for regression.

    Returns:
        clip: [C, T, H, W]
        target_phys: axial_resolution, not log-normalized
        video_path

    Target transform is handled in the training/evaluation loop:
        axial_resolution -> log -> normalize using train fold mean/std
    """

    def __init__(
        self,
        df: pd.DataFrame,
        crop_box,
        video_col="video_path",
        target_col="axial_resolution",
        clip_duration=2.6,
        num_frames=16,
        image_size=182,
        clips_per_video=1,
        random_start=True,
        seed=0,
        sampling_mode="contiguous",
        eps_sec=1.0 / 30.0,
        jitter=0.2,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.crop_box = crop_box
        self.video_col = video_col
        self.target_col = target_col

        self.clip_duration = float(clip_duration)
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.clips_per_video = int(clips_per_video)
        self.random_start = bool(random_start)
        self.seed = int(seed)

        self.sampling_mode = str(sampling_mode)

        if self.sampling_mode not in ("contiguous", "uniform"):
            raise ValueError(
                f"sampling_mode must be 'contiguous' or 'uniform', got {self.sampling_mode}"
            )

        self.eps_sec = float(eps_sec)
        self.jitter = float(jitter)

        self.video_paths = self.df[self.video_col].tolist()
        self.targets_phys = self.df[self.target_col].astype(float).tolist()

        # X3D / PyTorchVideo normalization
        self.mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)

    def __len__(self):
        return len(self.video_paths) * self.clips_per_video

    def _choose_start_sec(self, duration, video_idx, clip_idx):
        max_start = max(0.0, float(duration) - self.clip_duration)

        if max_start == 0.0:
            return 0.0

        if self.random_start:
            g = torch.Generator()
            g.manual_seed(self.seed + video_idx * 1000 + clip_idx)
            return float(torch.rand((), generator=g).item() * max_start)

        if self.clips_per_video == 1:
            return 0.0

        frac = clip_idx / (self.clips_per_video - 1)
        return float(frac * max_start)

    def _uniform_timestamps(self, duration, video_idx, clip_idx):
        """
        Deterministic or jittered uniform timestamps across full video.
        Useful for comparing against DINOv2-style full-video sampling.
        """

        t0 = 0.0
        t1 = max(0.0, float(duration) - 1e-3)

        if self.num_frames <= 1 or t1 <= t0:
            return [0.0]

        seg = (t1 - t0) / self.num_frames

        g = torch.Generator()
        g.manual_seed(self.seed + video_idx * 1000 + clip_idx)

        timestamps = []

        for i in range(self.num_frames):
            seg_start = t0 + i * seg
            seg_end = min(t1, seg_start + seg)

            if self.random_start and self.jitter > 0:
                span = max(1e-6, seg_end - seg_start)
                u = torch.rand((), generator=g).item()
                t = seg_start + u * span
            else:
                t = 0.5 * (seg_start + seg_end)

            timestamps.append(float(t))

        return timestamps

    def _spatial_process(self, clip):
        """
        Input:
            clip: [C, T, H, W]

        Output:
            clip: [C, T, image_size, image_size]
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
        video_idx = idx // self.clips_per_video
        clip_idx = idx % self.clips_per_video

        video_path = self.video_paths[video_idx]
        target_phys = float(self.targets_phys[video_idx])

        video = EncodedVideo.from_path(video_path)
        duration = float(video.duration)

        if self.sampling_mode == "uniform":
            timestamps = self._uniform_timestamps(
                duration=duration,
                video_idx=video_idx,
                clip_idx=clip_idx,
            )

            frames = []

            for t in timestamps:
                clip_dict = video.get_clip(
                    start_sec=float(t),
                    end_sec=float(min(duration, t + self.eps_sec)),
                )

                raw = clip_dict.get("video", None)

                if raw is None or raw.numel() == 0:
                    continue

                frames.append(raw[:, 0:1])

            if len(frames) == 0:
                raw_clip = video.get_clip(
                    start_sec=0.0,
                    end_sec=min(duration, self.clip_duration),
                )["video"]
            else:
                raw_clip = torch.cat(frames, dim=1)

            C, T_raw, H, W = raw_clip.shape

            if T_raw >= self.num_frames:
                raw_clip = raw_clip[:, :self.num_frames]
            else:
                last = raw_clip[:, -1:].repeat(
                    1,
                    self.num_frames - T_raw,
                    1,
                    1,
                )
                raw_clip = torch.cat([raw_clip, last], dim=1)

        else:
            start_sec = self._choose_start_sec(
                duration=duration,
                video_idx=video_idx,
                clip_idx=clip_idx,
            )

            end_sec = min(duration, start_sec + self.clip_duration)

            raw_clip = video.get_clip(
                start_sec=start_sec,
                end_sec=end_sec,
            )["video"]

            C, T_raw, H, W = raw_clip.shape

            if T_raw >= self.num_frames:
                indices = torch.linspace(
                    0,
                    T_raw - 1,
                    self.num_frames,
                ).long()
                raw_clip = raw_clip[:, indices]
            else:
                last = raw_clip[:, -1:].repeat(
                    1,
                    self.num_frames - T_raw,
                    1,
                    1,
                )
                raw_clip = torch.cat([raw_clip, last], dim=1)

        clip = self._spatial_process(raw_clip)

        return clip, torch.tensor(target_phys, dtype=torch.float32), video_path


# --------------------------------------------------
# Model
# --------------------------------------------------

def get_x3d_backbone(model_name: str, pretrained: bool = True):
    model_name = model_name.lower()

    if model_name == "x3d_xs":
        return x3d_xs(pretrained=pretrained)
    elif model_name == "x3d_s":
        return x3d_s(pretrained=pretrained)
    elif model_name == "x3d_m":
        return x3d_m(pretrained=pretrained)
    elif model_name == "x3d_l":
        return x3d_l(pretrained=pretrained)
    else:
        raise ValueError(
            f"Unknown X3D model_name={model_name}. "
            f"Use one of: x3d_xs, x3d_s, x3d_m, x3d_l"
        )


class X3DFinetuneRegressor(nn.Module):
    """
    X3D backbone + DINOv2-style regression head:

        Linear -> ReLU -> Dropout -> Linear
    """

    def __init__(
        self,
        model_name: str = "x3d_xs",
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.model_name = model_name

        self.backbone = get_x3d_backbone(
            model_name=model_name,
            pretrained=pretrained,
        )

        in_features = self.backbone.blocks[-1].proj.in_features

        # Remove classification projection.
        self.backbone.blocks[-1].proj = nn.Identity()

        self.regressor = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        feats = self.backbone(x)

        if feats.ndim > 2:
            feats = torch.flatten(feats, start_dim=1)

        pred = self.regressor(feats)

        return pred.squeeze(1)


def build_x3d_finetune_model(cfg: dict, device: torch.device):
    model = X3DFinetuneRegressor(
        model_name=cfg["model_name"],
        pretrained=cfg.get("pretrained", True),
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    )

    return model.to(device)


# --------------------------------------------------
# Freezing / fine-tuning utilities
# --------------------------------------------------

def freeze_backbone(model: X3DFinetuneRegressor):
    for p in model.backbone.parameters():
        p.requires_grad = False

    for p in model.regressor.parameters():
        p.requires_grad = True


def unfreeze_all(model: X3DFinetuneRegressor):
    for p in model.parameters():
        p.requires_grad = True


def unfreeze_last_n_blocks(model: X3DFinetuneRegressor, n_blocks: int = 1):
    """
    Freeze the whole backbone first, then unfreeze the last n X3D blocks.
    The regression head always remains trainable.
    """

    freeze_backbone(model)

    n_blocks = int(n_blocks)

    if n_blocks <= 0:
        return

    for block in model.backbone.blocks[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True

    for p in model.regressor.parameters():
        p.requires_grad = True


def get_trainable_parameter_groups(model: X3DFinetuneRegressor, cfg: dict):
    backbone_params = [
        p for p in model.backbone.parameters()
        if p.requires_grad
    ]

    head_params = [
        p for p in model.regressor.parameters()
        if p.requires_grad
    ]

    param_groups = []

    if len(backbone_params) > 0:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": cfg["lr_backbone"],
            }
        )

    if len(head_params) > 0:
        param_groups.append(
            {
                "params": head_params,
                "lr": cfg["lr_head"],
            }
        )

    return param_groups


# --------------------------------------------------
# Target transform
# --------------------------------------------------

def normalize_target_phys(
    target_phys: torch.Tensor,
    target_mean_t: torch.Tensor,
    target_std_t: torch.Tensor,
    use_log_target: bool = True,
):
    y = target_phys.float()

    if use_log_target:
        y = torch.log(y)

    y_norm = (y - target_mean_t) / target_std_t

    return y_norm


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
# Train / evaluate
# --------------------------------------------------

def run_train_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    target_mean_t,
    target_std_t,
    use_log_target=True,
    grad_clip=1.0,
):
    model.train()

    losses = []

    for videos, targets_phys, paths in loader:
        videos = videos.to(device, non_blocking=True)
        targets_phys = targets_phys.to(device, non_blocking=True)

        targets_norm = normalize_target_phys(
            target_phys=targets_phys,
            target_mean_t=target_mean_t,
            target_std_t=target_std_t,
            use_log_target=use_log_target,
        )

        optimizer.zero_grad()

        preds_norm = model(videos)

        loss = loss_fn(preds_norm, targets_norm)

        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(grad_clip),
            )

        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate_x3d_regression(
    model,
    loader,
    loss_fn,
    device,
    target_mean,
    target_std,
    target_mean_t,
    target_std_t,
    use_log_target=True,
):
    """
    Video-level evaluation.

    If clips_val > 1, predictions from multiple clips of the same video
    are averaged in normalized/log space before conversion to physical units.
    """

    model.eval()

    preds_per_video = defaultdict(list)
    targets_per_video = {}

    val_losses = []

    for videos, targets_phys, paths in loader:
        videos = videos.to(device, non_blocking=True)
        targets_phys = targets_phys.to(device, non_blocking=True)

        targets_norm = normalize_target_phys(
            target_phys=targets_phys,
            target_mean_t=target_mean_t,
            target_std_t=target_std_t,
            use_log_target=use_log_target,
        )

        preds_norm = model(videos)

        loss = loss_fn(preds_norm, targets_norm)
        val_losses.append(loss.item())

        preds_norm_np = preds_norm.detach().cpu().numpy()
        targets_phys_np = targets_phys.detach().cpu().numpy()

        for pred_n, target_p, path in zip(
            preds_norm_np,
            targets_phys_np,
            paths,
        ):
            preds_per_video[path].append(float(pred_n))
            targets_per_video[path] = float(target_p)

    rows = []

    preds_phys_all = []
    targets_phys_all = []

    for path, pred_list in preds_per_video.items():
        pred_norm_mean = float(np.mean(pred_list))
        true_phys = float(targets_per_video[path])

        pred_phys = float(
            inverse_transform_target(
                pred_norm_mean,
                target_mean,
                target_std,
                use_log_target,
            )
        )

        err = pred_phys - true_phys

        rows.append(
            {
                "video_path": path,
                "true_phys": true_phys,
                "pred_phys": pred_phys,
                "err_phys": err,
                "abs_err_phys": abs(err),
                "num_clips": len(pred_list),
            }
        )

        preds_phys_all.append(pred_phys)
        targets_phys_all.append(true_phys)

    if len(rows) == 0:
        metrics = {
            "val_loss": float("nan"),
            "val_mae_phys": float("nan"),
            "val_rmse_phys": float("nan"),
            "bias_phys": float("nan"),
        }
        pred_df = pd.DataFrame(rows)
        return metrics, pred_df

    preds_phys_all = np.array(preds_phys_all)
    targets_phys_all = np.array(targets_phys_all)

    mae = mean_absolute_error(targets_phys_all, preds_phys_all)
    mse = mean_squared_error(targets_phys_all, preds_phys_all)
    rmse = np.sqrt(mse)
    bias = np.mean(preds_phys_all - targets_phys_all)

    metrics = {
        "val_loss": float(np.mean(val_losses)),
        "val_mae_phys": float(mae),
        "val_rmse_phys": float(rmse),
        "bias_phys": float(bias),
    }

    pred_df = pd.DataFrame(rows)

    return metrics, pred_df


# --------------------------------------------------
# Main fold trainer
# --------------------------------------------------

def train_x3d_finetune_fold(
    fold: int,
    train_csv: str,
    val_csv: str,
    crop_box,
    cfg: dict,
    device: torch.device = None,
):
    """
    Train one X3D fold.

    Supports:
        frozen head-only training
        last-block fine-tuning
        last-N-block fine-tuning
        full fine-tuning

    Best checkpoint and early stopping are based on val_mae_phys.
    """

    cfg = dict(cfg)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_out_dir = os.path.join(cfg["out_dir"], f"fold_{fold}")
    os.makedirs(fold_out_dir, exist_ok=True)

    set_seed(cfg["seed"])

    train_df = pd.read_csv(train_csv).reset_index(drop=True)
    val_df = pd.read_csv(val_csv).reset_index(drop=True)

    # --------------------------------------------------
    # Target normalization from TRAIN fold only
    # Same as DINOv2
    # --------------------------------------------------

    y_train = train_df[cfg["target_col"]].astype(float).values

    if cfg["use_log_target"]:
        y_train = np.log(y_train)

    target_mean = float(y_train.mean())
    target_std = float(y_train.std() + 1e-8)

    target_mean_t = torch.tensor(target_mean, dtype=torch.float32, device=device)
    target_std_t = torch.tensor(target_std, dtype=torch.float32, device=device)

    print()
    print("=" * 80)
    print(f"X3D Fold {fold}")
    print("=" * 80)
    print(f"Model:       {cfg['model_name']}")
    print(f"Train CSV:   {train_csv}")
    print(f"Val CSV:     {val_csv}")
    print(f"Train videos:{len(train_df)}")
    print(f"Val videos:  {len(val_df)}")
    print(f"Sampling:    {cfg['sampling_mode']}")
    print(f"Target mean: {target_mean:.6f}")
    print(f"Target std:  {target_std:.6f}")
    print(f"Fine-tune:   {cfg['finetune_mode']}")

    # --------------------------------------------------
    # Datasets
    # --------------------------------------------------

    train_dataset = PrinterVideoDatasetX3DFinetune(
        df=train_df,
        crop_box=crop_box,
        video_col=cfg["video_col"],
        target_col=cfg["target_col"],
        clip_duration=cfg["clip_duration"],
        num_frames=cfg["num_frames"],
        image_size=cfg["image_size"],
        clips_per_video=cfg["clips_train"],
        random_start=True,
        seed=cfg["seed"],
        sampling_mode=cfg["sampling_mode"],
        eps_sec=cfg["eps_sec"],
        jitter=cfg["uniform_jitter"],
    )

    val_dataset = PrinterVideoDatasetX3DFinetune(
        df=val_df,
        crop_box=crop_box,
        video_col=cfg["video_col"],
        target_col=cfg["target_col"],
        clip_duration=cfg["clip_duration"],
        num_frames=cfg["num_frames"],
        image_size=cfg["image_size"],
        clips_per_video=cfg["clips_val"],
        random_start=False,
        seed=cfg["seed"],
        sampling_mode=cfg["sampling_mode"],
        eps_sec=cfg["eps_sec"],
        jitter=0.0,
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
    # Model
    # --------------------------------------------------

    model = build_x3d_finetune_model(cfg, device)

    loss_fn = torch.nn.SmoothL1Loss(beta=cfg["huber_beta"])

    best_val_mae_phys = float("inf")
    best_epoch = None
    best_metrics = None
    best_state = None

    history = []

    epochs_without_improvement = 0
    stopped_early = False
    stop_epoch = None
    global_epoch = 0

    # --------------------------------------------------
    # Phase 1: train head only
    # --------------------------------------------------

    if cfg["epochs_head"] > 0:
        print()
        print("Phase 1: frozen backbone, training regression head only")

        freeze_backbone(model)

        optimizer = torch.optim.AdamW(
            model.regressor.parameters(),
            lr=cfg["lr_head"],
            weight_decay=cfg["weight_decay"],
        )

        for epoch in range(1, cfg["epochs_head"] + 1):
            global_epoch += 1

            train_loss = run_train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                target_mean_t=target_mean_t,
                target_std_t=target_std_t,
                use_log_target=cfg["use_log_target"],
                grad_clip=cfg["grad_clip"],
            )

            val_metrics, val_pred_df = evaluate_x3d_regression(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                target_mean=target_mean,
                target_std=target_std,
                target_mean_t=target_mean_t,
                target_std_t=target_std_t,
                use_log_target=cfg["use_log_target"],
            )

            current_val_mae = val_metrics["val_mae_phys"]

            improved = current_val_mae < (
                best_val_mae_phys - cfg["min_delta"]
            )

            if improved:
                best_val_mae_phys = current_val_mae
                best_epoch = global_epoch
                best_metrics = deepcopy(val_metrics)
                best_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0

                val_pred_df.to_csv(
                    os.path.join(fold_out_dir, "val_predictions_best.csv"),
                    index=False,
                )
            else:
                epochs_without_improvement += 1

            row = {
                "fold": fold,
                "epoch": global_epoch,
                "phase": "head",
                "train_loss": train_loss,
                **val_metrics,
                "best_val_mae_phys_so_far": best_val_mae_phys,
                "improved": improved,
                "epochs_without_improvement": epochs_without_improvement,
            }

            history.append(row)

            print(
                f"Fold {fold} | "
                f"Head {epoch:03d}/{cfg['epochs_head']} | "
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
                stop_epoch = global_epoch
                print(
                    f"Early stopping during head phase. "
                    f"Best epoch: {best_epoch}, best val_mae_phys={best_val_mae_phys:.5f}"
                )
                break

    # --------------------------------------------------
    # Phase 2: fine-tuning
    # --------------------------------------------------

    if not stopped_early and cfg["epochs_ft"] > 0 and cfg["finetune_mode"] != "none":
        print()
        print("Phase 2: fine-tuning")

        mode = cfg["finetune_mode"]

        if mode == "full":
            unfreeze_all(model)

        elif mode == "last_block":
            unfreeze_last_n_blocks(model, n_blocks=1)

        elif mode == "last_n_blocks":
            unfreeze_last_n_blocks(
                model,
                n_blocks=cfg["last_n_blocks"],
            )

        else:
            raise ValueError(
                f"Unknown finetune_mode={mode}. "
                f"Use: none, last_block, last_n_blocks, full"
            )

        param_groups = get_trainable_parameter_groups(model, cfg)

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=cfg["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=cfg["sched_patience"],
        )

        for epoch in range(1, cfg["epochs_ft"] + 1):
            global_epoch += 1

            train_loss = run_train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                target_mean_t=target_mean_t,
                target_std_t=target_std_t,
                use_log_target=cfg["use_log_target"],
                grad_clip=cfg["grad_clip"],
            )

            val_metrics, val_pred_df = evaluate_x3d_regression(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                target_mean=target_mean,
                target_std=target_std,
                target_mean_t=target_mean_t,
                target_std_t=target_std_t,
                use_log_target=cfg["use_log_target"],
            )

            current_val_mae = val_metrics["val_mae_phys"]

            scheduler.step(current_val_mae)

            improved = current_val_mae < (
                best_val_mae_phys - cfg["min_delta"]
            )

            if improved:
                best_val_mae_phys = current_val_mae
                best_epoch = global_epoch
                best_metrics = deepcopy(val_metrics)
                best_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0

                val_pred_df.to_csv(
                    os.path.join(fold_out_dir, "val_predictions_best.csv"),
                    index=False,
                )

            else:
                epochs_without_improvement += 1

            row = {
                "fold": fold,
                "epoch": global_epoch,
                "phase": "finetune",
                "train_loss": train_loss,
                **val_metrics,
                "best_val_mae_phys_so_far": best_val_mae_phys,
                "improved": improved,
                "epochs_without_improvement": epochs_without_improvement,
            }

            history.append(row)

            print(
                f"Fold {fold} | "
                f"FT {epoch:03d}/{cfg['epochs_ft']} | "
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
                stop_epoch = global_epoch
                print(
                    f"Early stopping during fine-tuning. "
                    f"Best epoch: {best_epoch}, best val_mae_phys={best_val_mae_phys:.5f}"
                )
                break

    # --------------------------------------------------
    # Save best model
    # --------------------------------------------------

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "fold": fold,
        "epoch": best_epoch,
        "model_state_dict": model.state_dict(),
        "cfg": cfg,
        "target_mean": target_mean,
        "target_std": target_std,
        "best_val_mae_phys": best_val_mae_phys,
        "best_metrics": best_metrics,
    }

    torch.save(
        checkpoint,
        os.path.join(fold_out_dir, "model_best.pth"),
    )

    # --------------------------------------------------
    # Save history
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
        "target_mean": target_mean,
        "target_std": target_std,
        "num_train": len(train_df),
        "num_val": len(val_df),
        "crop_box": crop_box,
        "model_name": cfg["model_name"],
        "sampling_mode": cfg["sampling_mode"],
        "clips_train": cfg["clips_train"],
        "clips_val": cfg["clips_val"],
        "num_frames": cfg["num_frames"],
        "image_size": cfg["image_size"],
        "finetune_mode": cfg["finetune_mode"],
        "last_n_blocks": cfg["last_n_blocks"],
        "stopped_early": stopped_early,
        "stop_epoch": stop_epoch,
        "early_stopping_patience": cfg["early_stopping_patience"],
        "min_delta": cfg["min_delta"],
        "train_csv": train_csv,
        "val_csv": val_csv,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(
        os.path.join(fold_out_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(fold_summary, f, indent=2)

    return history_df, fold_summary
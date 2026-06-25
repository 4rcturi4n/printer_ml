import os
import json
import random
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# Mamba backend
#   On the server: pip install mamba-ssm causal-conv1d
#   Falls back to a pure-PyTorch implementation if not available
# --------------------------------------------------
try:
    from mamba_ssm import Mamba as _MambaSSM
    _MAMBA_SSM_AVAILABLE = True
except ImportError:
    _MAMBA_SSM_AVAILABLE = False


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Frame loader (same logic as dinov2_regression)
# --------------------------------------------------

def sample_video_frames(
    video_path: str,
    num_frames: int = 32,
    image_size: int = 224,
    crop_box=None,
):
    """
    Uniformly samples num_frames from a video.
    Returns [T, 3, H, W] float32 tensor, ImageNet-normalized.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {video_path}")

    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    mean_arr = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std_arr  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if crop_box is not None:
            x1, y1, x2, y2 = crop_box
            h, w = frame.shape[:2]
            x1 = max(0, min(int(x1), w))
            x2 = max(0, min(int(x2), w))
            y1 = max(0, min(int(y1), h))
            y2 = max(0, min(int(y2), h))
            frame = frame[y1:y2, x1:x2]

        frame = cv2.resize(frame, (image_size, image_size))
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - mean_arr) / std_arr
        frame = np.transpose(frame, (2, 0, 1))  # [3, H, W]
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No readable frames: {video_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1])

    return torch.tensor(np.stack(frames[:num_frames], axis=0), dtype=torch.float32)


# --------------------------------------------------
# Dataset — returns full frame sequence per video
# --------------------------------------------------

class FrameSequenceDataset(Dataset):
    """
    Returns:
        frames       [T, 3, H, W]
        target_norm  scalar (normalized)
        target_phys  scalar (physical μm)
        video_id     str
    """

    def __init__(
        self,
        df: pd.DataFrame,
        video_col: str,
        target_col: str,
        num_frames: int,
        image_size: int,
        target_mean: float,
        target_std: float,
        use_log_target: bool = True,
        crop_box=None,
    ):
        self.df           = df.reset_index(drop=True)
        self.video_col    = video_col
        self.target_col   = target_col
        self.num_frames   = num_frames
        self.image_size   = image_size
        self.target_mean  = target_mean
        self.target_std   = target_std
        self.use_log_target = use_log_target
        self.crop_box     = crop_box

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        video_path = row[self.video_col]
        phys       = float(row[self.target_col])

        frames = sample_video_frames(
            video_path  = video_path,
            num_frames  = self.num_frames,
            image_size  = self.image_size,
            crop_box    = self.crop_box,
        )

        y = np.log(phys) if self.use_log_target else phys
        y_norm = (y - self.target_mean) / self.target_std

        video_id = row["video_id"] if "video_id" in row.index else os.path.basename(video_path)

        return (
            frames,
            torch.tensor(y_norm, dtype=torch.float32),
            torch.tensor(phys,   dtype=torch.float32),
            str(video_id),
        )


# --------------------------------------------------
# Mamba SSM layer
#
# When mamba_ssm is installed (server with CUDA):
#   uses the optimised fused CUDA kernels from the official library.
#
# Fallback (no install / CPU):
#   pure-PyTorch sequential scan — same maths, slower but correct.
# --------------------------------------------------

class _MambaLayerFallback(nn.Module):
    """Pure-PyTorch Mamba block. Identical maths, no custom kernels."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True,
        )
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj  = nn.Linear(1, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        xz            = self.in_proj(x)
        x_inner, z    = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_inner.permute(0, 2, 1))[:, :, :T].permute(0, 2, 1)
        x_conv = F.silu(x_conv)
        ssm    = self.x_proj(x_conv)
        dt_raw, B_p, C_p = ssm[..., :1], ssm[..., 1:self.d_state+1], ssm[..., self.d_state+1:]
        dt = F.softplus(self.dt_proj(dt_raw))
        A  = -torch.exp(self.A_log.float())
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = dt.unsqueeze(-1) * B_p.unsqueeze(2)
        h, ys = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype), []
        for t in range(T):
            h = dA[:, t] * h + dB[:, t] * x_conv[:, t].unsqueeze(-1)
            ys.append((h * C_p[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + x_conv * self.D
        return self.out_proj(y * F.silu(z))


def _build_mamba_layer(d_model: int, d_state: int, d_conv: int, expand: int) -> nn.Module:
    """Returns the best available Mamba layer for this environment."""
    if _MAMBA_SSM_AVAILABLE:
        return _MambaSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    return _MambaLayerFallback(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


# --------------------------------------------------
# Sequence head: stacked Mamba layers + per-timestep MLP head
#
# Pulled out of MambaEarlyPredictor so it can be reused directly on
# pre-cached DINOv2 embeddings (no backbone, no video I/O) — see
# mamba_cached_regression.py.
# --------------------------------------------------

class MambaSequenceRegressor(nn.Module):
    """
    1. Stacked Mamba layers process an embedding sequence causally → [B, T, D]
    2. MLP head predicts axial resolution at every timestep → [B, T]

    Takes already-encoded embeddings [B, T, D] as input (e.g. cached
    frozen-DINOv2 output) — no image encoder inside this module.
    """

    def __init__(
        self,
        embed_dim:      int,
        n_mamba_layers: int = 2,
        d_state:        int = 16,
        d_conv:         int = 4,
        expand:         int = 2,
        hidden_dim:     int = 256,
        dropout:        float = 0.2,
    ):
        super().__init__()

        # residual Mamba blocks with pre-norm
        # automatically uses mamba_ssm CUDA kernels when available
        self.mamba_blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm":  nn.LayerNorm(embed_dim),
                "mamba": _build_mamba_layer(embed_dim, d_state=d_state, d_conv=d_conv, expand=expand),
            })
            for _ in range(n_mamba_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D] embeddings
        returns: [B, T]  — prediction at every timestep
        """
        for block in self.mamba_blocks:
            x = x + block["mamba"](block["norm"](x))

        preds = self.head(x).squeeze(-1)   # [B, T]
        return preds


# --------------------------------------------------
# Full model: DINOv2 per-frame + Mamba + per-timestep head
# --------------------------------------------------

class MambaEarlyPredictor(nn.Module):
    """
    1. DINOv2 (frozen) encodes each frame independently → [B, T, D]
    2. Stacked Mamba layers process the sequence causally → [B, T, D]
    3. MLP head predicts axial resolution at every timestep → [B, T]

    At inference time you can stop at any frame and read a prediction.
    """

    def __init__(
        self,
        dinov2_name:    str = "dinov2_vits14",
        n_mamba_layers: int = 2,
        d_state:        int = 16,
        d_conv:         int = 4,
        expand:         int = 2,
        hidden_dim:     int = 256,
        dropout:        float = 0.2,
    ):
        super().__init__()

        self.backbone = torch.hub.load("facebookresearch/dinov2", dinov2_name)
        for p in self.backbone.parameters():
            p.requires_grad = False

        embed_dim = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }[dinov2_name]

        self.embed_dim = embed_dim

        self.sequence_regressor = MambaSequenceRegressor(
            embed_dim=embed_dim,
            n_mamba_layers=n_mamba_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    @torch.no_grad()
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B, T, 3, H, W] → [B, T, D]"""
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)
        emb  = self.backbone(flat)
        return emb.view(B, T, -1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: [B, T, 3, H, W]
        returns: [B, T]  — prediction at every timestep
        """
        x = self.encode_frames(frames)
        return self.sequence_regressor(x)


# --------------------------------------------------
# Loss: ramp-weighted per-timestep SmoothL1
# --------------------------------------------------

def ramp_weighted_loss(
    preds:   torch.Tensor,
    targets: torch.Tensor,
    loss_fn,
) -> torch.Tensor:
    """
    preds:   [B, T]
    targets: [B]

    Weight ramps linearly from 1/T at t=0 to 1.0 at t=T-1.
    Early frames contribute little; the model is held fully accountable by the end.
    """
    B, T   = preds.shape
    weights = torch.linspace(1.0 / T, 1.0, T, device=preds.device)
    target_exp = targets.unsqueeze(1).expand_as(preds)

    total = torch.tensor(0.0, device=preds.device)
    for t in range(T):
        total = total + weights[t] * loss_fn(preds[:, t], target_exp[:, t])

    return total / T


# --------------------------------------------------
# Target inverse transform
# --------------------------------------------------

def inverse_transform(
    y_norm,
    target_mean: float,
    target_std:  float,
    use_log_target: bool = True,
):
    y = np.asarray(y_norm) * target_std + target_mean
    if use_log_target:
        y = np.exp(y)
    return y


# --------------------------------------------------
# Evaluation — per-timestep MAE + EPT
# --------------------------------------------------

@torch.no_grad()
def evaluate_early_prediction(
    model:           nn.Module,
    loader:          DataLoader,
    device:          torch.device,
    target_mean:     float,
    target_std:      float,
    use_log_target:  bool,
    ept_threshold_um: float,
):
    """
    Runs the model on the full validation set and computes:

    1. MAE / RMSE at every frame position (mae_per_t_df)
    2. Per-video prediction trajectory (per_video_df)
    3. EPT — first frame where the prediction error stays below ept_threshold_um
       for all remaining frames, reported as % of video duration (ept_df)

    Returns:
        metrics       dict  — scalar summary stats
        per_video_df  DataFrame
        mae_per_t_df  DataFrame
        ept_df        DataFrame
    """
    model.eval()

    all_preds_norm   = []
    all_targets_phys = []
    all_video_ids    = []

    for frames, targets_norm, targets_phys, video_ids in loader:
        frames = frames.to(device)
        preds  = model(frames)                  # [B, T]
        all_preds_norm.append(preds.cpu().numpy())
        all_targets_phys.append(targets_phys.numpy())
        all_video_ids.extend(list(video_ids))

    preds_norm_arr   = np.concatenate(all_preds_norm,   axis=0)   # [N, T]
    targets_phys_arr = np.concatenate(all_targets_phys, axis=0)   # [N]
    N, T = preds_norm_arr.shape

    # physical predictions at every timestep
    preds_phys_arr = np.stack([
        inverse_transform(preds_norm_arr[:, t], target_mean, target_std, use_log_target)
        for t in range(T)
    ], axis=1)                                                      # [N, T]

    frame_pct = np.linspace(0.0, 100.0, T)

    # --------------------------------------------------
    # 1. MAE / RMSE per timestep
    # --------------------------------------------------
    mae_per_t  = [mean_absolute_error(targets_phys_arr, preds_phys_arr[:, t]) for t in range(T)]
    rmse_per_t = [np.sqrt(mean_squared_error(targets_phys_arr, preds_phys_arr[:, t])) for t in range(T)]

    mae_per_t_df = pd.DataFrame({
        "frame_idx":  np.arange(T),
        "frame_pct":  frame_pct,
        "mae_phys":   mae_per_t,
        "rmse_phys":  rmse_per_t,
    })

    # --------------------------------------------------
    # 2. Final-frame scalar metrics
    # --------------------------------------------------
    final_mae  = float(mae_per_t[-1])
    final_rmse = float(rmse_per_t[-1])
    final_bias = float(np.mean(preds_phys_arr[:, -1] - targets_phys_arr))

    # --------------------------------------------------
    # 3. EPT per video
    #    First frame t where |pred_t - true| < threshold for ALL t' >= t
    # --------------------------------------------------
    ept_pcts  = []
    ept_found = []

    for i in range(N):
        errors    = np.abs(preds_phys_arr[i] - targets_phys_arr[i])
        ept_frame = None
        for t in range(T):
            if np.all(errors[t:] < ept_threshold_um):
                ept_frame = t
                break
        if ept_frame is not None:
            ept_pcts.append(float(frame_pct[ept_frame]))
            ept_found.append(True)
        else:
            ept_pcts.append(100.0)
            ept_found.append(False)

    # --------------------------------------------------
    # 4. Per-video trajectory DataFrame
    # --------------------------------------------------
    rows = []
    for i, vid_id in enumerate(all_video_ids):
        for t in range(T):
            rows.append({
                "video_id":     vid_id,
                "frame_idx":    t,
                "frame_pct":    float(frame_pct[t]),
                "pred_phys":    float(preds_phys_arr[i, t]),
                "true_phys":    float(targets_phys_arr[i]),
                "abs_err_phys": float(abs(preds_phys_arr[i, t] - targets_phys_arr[i])),
                "ept_pct":      float(ept_pcts[i]),
                "ept_found":    bool(ept_found[i]),
            })
    per_video_df = pd.DataFrame(rows)

    # --------------------------------------------------
    # 5. EPT summary per video
    # --------------------------------------------------
    ept_df = pd.DataFrame({
        "video_id":        all_video_ids,
        "true_phys":       targets_phys_arr,
        "final_pred_phys": preds_phys_arr[:, -1],
        "final_abs_err":   np.abs(preds_phys_arr[:, -1] - targets_phys_arr),
        "ept_pct":         ept_pcts,
        "ept_found":       ept_found,
    })

    metrics = {
        "final_mae_phys":          final_mae,
        "final_rmse_phys":         final_rmse,
        "final_bias_phys":         final_bias,
        "mean_ept_pct":            float(np.mean(ept_pcts)),
        "pct_videos_ept_found":    float(np.mean(ept_found) * 100.0),
        "ept_threshold_um":        float(ept_threshold_um),
        "n_videos":                N,
        "n_frames":                T,
    }

    return metrics, per_video_df, mae_per_t_df, ept_df


# --------------------------------------------------
# Plots
# --------------------------------------------------

def plot_mae_curve(mae_per_t_df: pd.DataFrame, out_path: str, fold: int):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mae_per_t_df["frame_pct"], mae_per_t_df["mae_phys"],
            label="MAE (μm)", color="steelblue", linewidth=2)
    ax.plot(mae_per_t_df["frame_pct"], mae_per_t_df["rmse_phys"],
            label="RMSE (μm)", color="tomato", linestyle="--", linewidth=2)
    ax.set_xlabel("Video progress (%)")
    ax.set_ylabel("Error (μm)")
    ax.set_title(f"Fold {fold} — Prediction error vs. video progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ept_distribution(ept_df: pd.DataFrame, out_path: str, fold: int, ept_threshold_um: float):
    found = ept_df[ept_df["ept_found"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(found) > 0:
        ax.hist(found["ept_pct"], bins=min(10, len(found)), color="steelblue", edgecolor="white")
    ax.set_xlabel("EPT — video progress when prediction stabilised (%)")
    ax.set_ylabel("Number of videos")
    ax.set_title(
        f"Fold {fold} — Early Prediction Time distribution\n"
        f"(threshold = {ept_threshold_um} μm, "
        f"{len(found)}/{len(ept_df)} videos detected)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_video_trajectories(
    per_video_df: pd.DataFrame,
    out_dir: str,
    ept_threshold_um: float,
    max_videos: int = 10,
):
    os.makedirs(out_dir, exist_ok=True)
    video_ids = per_video_df["video_id"].unique()[:max_videos]

    for vid_id in video_ids:
        sub      = per_video_df[per_video_df["video_id"] == vid_id].sort_values("frame_idx")
        true_val = float(sub["true_phys"].iloc[0])
        ept_pct  = float(sub["ept_pct"].iloc[0])
        detected = bool(sub["ept_found"].iloc[0])

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(sub["frame_pct"], sub["pred_phys"],
                color="steelblue", linewidth=2, label="Predicted axial res.")
        ax.axhline(true_val, color="tomato", linestyle="--", linewidth=1.5,
                   label=f"True: {true_val:.3f} μm")
        ax.axhspan(true_val - ept_threshold_um, true_val + ept_threshold_um,
                   alpha=0.1, color="tomato", label=f"±{ept_threshold_um} μm band")
        if detected:
            ax.axvline(ept_pct, color="green", linestyle=":", linewidth=2,
                       label=f"EPT: {ept_pct:.1f}% of video")
        ax.set_xlabel("Video progress (%)")
        ax.set_ylabel("Axial resolution (μm)")
        ax.set_title(f"{vid_id}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        safe_id = str(vid_id).replace("/", "_").replace("\\", "_").replace(" ", "_")
        fig.savefig(os.path.join(out_dir, f"{safe_id}_trajectory.png"), dpi=120)
        plt.close(fig)


# --------------------------------------------------
# Train one fold
# --------------------------------------------------

def train_mamba_fold(
    fold:     int,
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    cfg:      dict,
    device:   torch.device,
):
    print()
    print("=" * 80)
    print(f"Mamba Early Predictor — Fold {fold}")
    print("=" * 80)

    fold_out_dir = os.path.join(cfg["out_dir"], f"fold_{fold}")
    os.makedirs(fold_out_dir, exist_ok=True)

    # target normalisation from train fold only
    y_train = train_df[cfg["target_col"]].astype(float).values
    if cfg["use_log_target"]:
        y_train = np.log(y_train)
    target_mean = float(y_train.mean())
    target_std  = float(y_train.std() + 1e-8)

    print(f"Train videos : {len(train_df)}")
    print(f"Val videos   : {len(val_df)}")
    print(f"Target mean  : {target_mean:.6f}")
    print(f"Target std   : {target_std:.6f}")
    print(f"EPT threshold: {cfg['ept_threshold_um']} μm")

    train_dataset = FrameSequenceDataset(
        df=train_df, video_col=cfg["video_col"], target_col=cfg["target_col"],
        num_frames=cfg["num_frames"], image_size=cfg["image_size"],
        target_mean=target_mean, target_std=target_std,
        use_log_target=cfg["use_log_target"], crop_box=cfg["crop_box"],
    )
    val_dataset = FrameSequenceDataset(
        df=val_df, video_col=cfg["video_col"], target_col=cfg["target_col"],
        num_frames=cfg["num_frames"], image_size=cfg["image_size"],
        target_mean=target_mean, target_std=target_std,
        use_log_target=cfg["use_log_target"], crop_box=cfg["crop_box"],
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=cfg["num_workers"], pin_memory=pin)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=cfg["num_workers"], pin_memory=pin)

    model = MambaEarlyPredictor(
        dinov2_name    = cfg["dinov2_model"],
        n_mamba_layers = cfg["n_mamba_layers"],
        d_state        = cfg["d_state"],
        d_conv         = cfg["d_conv"],
        expand         = cfg["expand"],
        hidden_dim     = cfg["hidden_dim"],
        dropout        = cfg["dropout"],
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn   = nn.SmoothL1Loss()

    best_val_mae  = float("inf")
    best_epoch    = None
    best_metrics  = None
    history       = []
    epochs_no_imp = 0
    stopped_early = False
    stop_epoch    = None

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_losses = []

        for frames, targets_norm, targets_phys, video_ids in train_loader:
            frames       = frames.to(device)
            targets_norm = targets_norm.to(device)

            optimizer.zero_grad()
            preds = model(frames)                                    # [B, T]
            loss  = ramp_weighted_loss(preds, targets_norm, loss_fn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        val_metrics, _, mae_per_t_df, ept_df = evaluate_early_prediction(
            model=model, loader=val_loader, device=device,
            target_mean=target_mean, target_std=target_std,
            use_log_target=cfg["use_log_target"],
            ept_threshold_um=cfg["ept_threshold_um"],
        )

        current_mae = val_metrics["final_mae_phys"]
        improved    = current_mae < (best_val_mae - cfg["min_delta"])

        if improved:
            best_val_mae  = current_mae
            best_epoch    = epoch
            best_metrics  = deepcopy(val_metrics)
            epochs_no_imp = 0

            torch.save({
                "fold": fold, "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "cfg": cfg,
                "target_mean": target_mean, "target_std": target_std,
                "best_val_mae_phys": best_val_mae,
                "best_metrics": best_metrics,
            }, os.path.join(fold_out_dir, "model_best.pth"))

            mae_per_t_df.to_csv(os.path.join(fold_out_dir, "mae_per_timestep.csv"), index=False)
            ept_df.to_csv(os.path.join(fold_out_dir, "ept_summary.csv"), index=False)
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
            f"Fold {fold} | Epoch {epoch:03d} | "
            f"train_loss={train_loss:.5f} | "
            f"val_mae={current_mae:.5f} | "
            f"mean_ept={val_metrics['mean_ept_pct']:.1f}% | "
            f"best={best_val_mae:.5f} | "
            f"no_improve={epochs_no_imp}/{cfg['early_stopping_patience']}"
        )

        if epochs_no_imp >= cfg["early_stopping_patience"]:
            stopped_early = True
            stop_epoch    = epoch
            print(f"\nEarly stopping fold {fold} at epoch {epoch}. Best epoch: {best_epoch}, MAE: {best_val_mae:.5f}")
            break

    pd.DataFrame(history).to_csv(os.path.join(fold_out_dir, "history.csv"), index=False)

    # --------------------------------------------------
    # Re-evaluate best checkpoint and save all outputs
    # --------------------------------------------------
    ckpt = torch.load(os.path.join(fold_out_dir, "model_best.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    best_val_metrics, per_video_df, mae_per_t_df, ept_df = evaluate_early_prediction(
        model=model, loader=val_loader, device=device,
        target_mean=target_mean, target_std=target_std,
        use_log_target=cfg["use_log_target"],
        ept_threshold_um=cfg["ept_threshold_um"],
    )

    per_video_df.to_csv(os.path.join(fold_out_dir, "val_predictions_per_timestep.csv"), index=False)
    mae_per_t_df.to_csv(os.path.join(fold_out_dir, "mae_per_timestep.csv"), index=False)
    ept_df.to_csv(os.path.join(fold_out_dir, "ept_summary.csv"), index=False)

    plot_mae_curve(mae_per_t_df, os.path.join(fold_out_dir, "mae_curve.png"), fold)
    plot_ept_distribution(
        ept_df, os.path.join(fold_out_dir, "ept_distribution.png"),
        fold, cfg["ept_threshold_um"],
    )
    plot_video_trajectories(
        per_video_df,
        out_dir=os.path.join(fold_out_dir, "monitoring_plots"),
        ept_threshold_um=cfg["ept_threshold_um"],
        max_videos=cfg.get("max_trajectory_plots", 10),
    )

    fold_summary = {
        "fold":                  fold,
        "best_epoch":            best_epoch,
        "best_val_mae_phys":     float(best_val_mae),
        "best_val_rmse_phys":    float(best_val_metrics["final_rmse_phys"]),
        "best_bias_phys":        float(best_val_metrics["final_bias_phys"]),
        "mean_ept_pct":          float(best_val_metrics["mean_ept_pct"]),
        "pct_videos_ept_found":  float(best_val_metrics["pct_videos_ept_found"]),
        "ept_threshold_um":      float(cfg["ept_threshold_um"]),
        "stopped_early":         stopped_early,
        "stop_epoch":            stop_epoch,
        "target_mean":           target_mean,
        "target_std":            target_std,
        "num_train":             len(train_df),
        "num_val":               len(val_df),
        "history_csv":           os.path.join(fold_out_dir, "history.csv"),
        "model_best":            os.path.join(fold_out_dir, "model_best.pth"),
        "mae_per_timestep_csv":  os.path.join(fold_out_dir, "mae_per_timestep.csv"),
        "ept_summary_csv":       os.path.join(fold_out_dir, "ept_summary.csv"),
        "per_video_csv":         os.path.join(fold_out_dir, "val_predictions_per_timestep.csv"),
    }

    with open(os.path.join(fold_out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)

    return pd.DataFrame(history), fold_summary

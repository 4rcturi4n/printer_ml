import os, json, time, random
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.models.hub import x3d_xs
import matplotlib.pyplot as plt


# ---------- helpers ----------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def fmt(x):
    if isinstance(x, float):
        s = f"{x:.6g}"
        return s.replace(".", "p")
    return str(x)

def make_run_name(cfg: dict) -> str:
    keys = [
        "model_name", "freeze", "head_dropout",
        "lr_head", "lr_backbone", "weight_decay",
        "huber_beta",
        "clip_duration", "num_frames", "image_size",
        "clips_train", "clips_val", "batch_size",
        "seed",
        "sampling_mode"
    ]
    return "__".join([f"{k}-{fmt(cfg[k])}" for k in keys if k in cfg])


def append_results_row(results_csv: str, row: dict):
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    df_row = pd.DataFrame([row])
    if os.path.exists(results_csv):
        df_row.to_csv(results_csv, mode="a", header=False, index=False)
    else:
        df_row.to_csv(results_csv, mode="w", header=True, index=False)


# ---------- dataset ----------
class PrinterVideoDatasetX3DReg(torch.utils.data.Dataset):
    """
    Two sampling modes:
      - "contiguous": random contiguous clip of clip_duration
      - "uniform": low temporal frequency sampling (num_frames timestamps across full video)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        crop_box,
        clip_duration=2.6,            #duration in seconds for contiguous clip mode
        num_frames=24,                #how many frame the final clip should have
        image_size=182,               #frames are resized for X3D to 182x182
        clips_per_video=1,            #1 for uniform, 3 for contiguous 
        random_start=True,            #whether to sample randomly
        seed=0,
        target_col="log_axial_resolution",

        
        sampling_mode="contiguous",   # "contiguous" or "uniform"
        eps_sec=1.0/30.0,             # small window to fetch a frame at timestamp 
        jitter=0.2,                   # only used for uniform mode when random_start=True
    ):
        super().__init__()
        self.crop_box = crop_box
        self.clip_duration = float(clip_duration)
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.clips_per_video = int(clips_per_video)
        self.random_start = bool(random_start)
        self.seed = int(seed)

        self.sampling_mode = str(sampling_mode)
        if self.sampling_mode not in ("contiguous", "uniform"):
            raise ValueError(f"sampling_mode must be 'contiguous' or 'uniform', got {self.sampling_mode}")
        self.eps_sec = float(eps_sec)
        self.jitter = float(jitter)

        self.video_paths = df["video_path"].tolist()
        self.y = df[target_col].astype(float).tolist()

        mean = [0.45, 0.45, 0.45]
        std  = [0.225, 0.225, 0.225]
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std  = torch.tensor(std).view(3, 1, 1, 1)

    def __len__(self):
        #return the final length of the dataset after sampling
        return len(self.video_paths) * self.clips_per_video

    def _choose_start_sec(self, duration, video_idx, clip_idx):
        #only used for contiguous mode
        
        max_start = max(0.0, float(duration) - self.clip_duration)        # For the case clip_duration> video_duration
        if max_start == 0.0:                                              
            return 0.0
        if self.random_start:                                             # If random_start=True make reproducible randomness
            g = torch.Generator()
            g.manual_seed(self.seed + video_idx * 1000 + clip_idx)
            return float(torch.rand((), generator=g).item() * max_start)
        else:                                                             # If random_start=False 
            if self.clips_per_video == 1:
                return 0.0
            frac = clip_idx / (self.clips_per_video - 1)
            return float(frac * max_start)

    def _uniform_timestamps(self, duration, video_idx, clip_idx):
        # Uniform low tf sampling
        # Sample num_frames timestamps across the whole video in chronological order.
        # Avoid sampling exactly at the end
        t0 = 0.0                                                          # Start of the video
        t1 = max(0.0, float(duration) - 1e-3)                             # End of the video (little before end)

        if self.num_frames <= 1 or t1 <= t0:                              # If some weird case happens return timestamp at 0
            return [0.0]

        seg = (t1 - t0) / self.num_frames                                 # Divide the video into equal segments to sample a frame from each segment

        g = torch.Generator()
        g.manual_seed(self.seed + video_idx * 1000 + clip_idx)

        ts = []
        for i in range(self.num_frames):
            seg_start = t0 + i * seg
            seg_end = min(t1, seg_start + seg)

            if self.random_start and self.jitter > 0:
                # Sample within the segment (jittered). jitter just controls whether we randomize or not;
                # the segment itself already limits the randomness.
                span = max(1e-6, seg_end - seg_start)
                u = torch.rand((), generator=g).item()
                t = seg_start + u * span
            else:
                # Deterministic: center of segment
                t = 0.5 * (seg_start + seg_end)

            ts.append(float(t))

        return ts

    def _spatial_process(self, clip):
        x1, y1, x2, y2 = self.crop_box
        clip = clip[:, :, y1:y2, x1:x2]

        clip = clip.permute(1, 0, 2, 3)  # (T,C,H,W)
        clip = F.interpolate(
            clip,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        clip = clip.permute(1, 0, 2, 3)  # (C,T,H,W)

        clip = clip.to(torch.float32) / 255.0
        clip = (clip - self.mean) / self.std
        return clip

    def __getitem__(self, idx):
        video_idx = idx // self.clips_per_video
        clip_idx  = idx % self.clips_per_video

        video_path = self.video_paths[video_idx]
        y = float(self.y[video_idx])

        video = EncodedVideo.from_path(video_path)
        duration = float(video.duration)

        # ----------------- TEMPORAL SAMPLING -----------------
        if self.sampling_mode == "uniform":
            # Low temporal frequency sampling across whole video
            ts = self._uniform_timestamps(duration, video_idx, clip_idx)
            frames = []
            eps = self.eps_sec

            for t in ts:
                clip_dict = video.get_clip(start_sec=float(t), end_sec=float(min(duration, t + eps)))
                v = clip_dict.get("video", None)  # (C,T,H,W)
                if v is None or v.numel() == 0:
                    continue
                # take first frame from this tiny clip
                frames.append(v[:, 0:1])  # (C,1,H,W)

            if len(frames) == 0:
                # Fallback: take a small contiguous clip from the start
                raw_clip = video.get_clip(0.0, min(duration, self.clip_duration))["video"]
            else:
                raw_clip = torch.cat(frames, dim=1)  # (C,T,H,W)

            # ensure exactly num_frames
            C, T_raw, H, W = raw_clip.shape
            if T_raw >= self.num_frames:
                raw_clip = raw_clip[:, :self.num_frames]
            else:
                last_frame = raw_clip[:, -1:].repeat(1, self.num_frames - T_raw, 1, 1)
                raw_clip = torch.cat([raw_clip, last_frame], dim=1)

        else:
            # Original contiguous clip behavior
            start_sec = self._choose_start_sec(duration, video_idx, clip_idx)
            end_sec = min(duration, start_sec + self.clip_duration)

            raw_clip = video.get_clip(start_sec=start_sec, end_sec=end_sec)["video"]  # (C,T,H,W)

            C, T_raw, H, W = raw_clip.shape
            if T_raw >= self.num_frames:
                indices = torch.linspace(0, T_raw - 1, self.num_frames).long()
                raw_clip = raw_clip[:, indices]
            else:
                last_frame = raw_clip[:, -1:].repeat(1, self.num_frames - T_raw, 1, 1)
                raw_clip = torch.cat([raw_clip, last_frame], dim=1)

        clip = self._spatial_process(raw_clip)
        return clip, y, video_path


# ---------- model ----------
class X3DRegressor(nn.Module):
    def __init__(self, backbone: nn.Module, in_features: int, dropout=0.5):
        super().__init__()
        self.backbone = backbone
        self.backbone.blocks[-1].proj = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

def build_model_reg(cfg: dict, device: torch.device) -> nn.Module:
    backbone = x3d_xs(pretrained=True)
    in_features = backbone.blocks[-1].proj.in_features
    model = X3DRegressor(backbone, in_features=in_features, dropout=float(cfg["head_dropout"]))
    return model.to(device)

def freeze_backbone(model: X3DRegressor):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

def unfreeze_all(model: X3DRegressor):
    for p in model.parameters():
        p.requires_grad = True


# ---------- training ----------
class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.bad = 0
        self.best_state = None

    def step(self, val_loss, model):
        val_loss = float(val_loss)
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.bad = 0
            self.best_state = deepcopy(model.state_dict())
            return False
        self.bad += 1
        return self.bad >= self.patience

def run_epoch_reg(model, loader, optimizer, criterion, device, train_mode, y_mean_t, y_std_t):
    model.train() if train_mode else model.eval()
    running_loss, total = 0.0, 0

    for videos, y, _ in loader:
        videos = videos.to(device)
        y = y.to(device).float().view(-1, 1)
        y_norm = (y - y_mean_t) / y_std_t

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            pred_norm = model(videos).view(-1, 1)
            loss = criterion(pred_norm, y_norm)
            if train_mode:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        running_loss += loss.item() * videos.size(0)
        total += videos.size(0)

    return running_loss / max(total, 1)

@torch.no_grad()
def video_level_eval_reg(model, loader, device, y_mean, y_std):
    model.eval()
    preds_per_video = defaultdict(list)
    true_per_video = {}

    for videos, y, paths in loader:
        videos = videos.to(device)
        pred_norm = model(videos).view(-1).cpu().numpy()
        y_true_log = y.cpu().numpy().astype(float)

        pred_log = pred_norm * y_std + y_mean

        for p, t, vp in zip(pred_log, y_true_log, paths):
            preds_per_video[vp].append(float(p))
            true_per_video[vp] = float(t)

    abs_err_phys, sq_err_phys, err_phys = [], [], []
    rows = []

    for vp, plist in preds_per_video.items():
        pred_log_mean = float(np.mean(plist))
        true_log = float(true_per_video[vp])

        pred_phys = float(np.exp(pred_log_mean))
        true_phys = float(np.exp(true_log))

        e = pred_phys - true_phys
        err_phys.append(e)
        abs_err_phys.append(abs(e))
        sq_err_phys.append(e * e)

        rows.append({
            "video_path": vp,
            "true_phys": true_phys,
            "pred_phys": pred_phys,
            "err_phys": e,
            "abs_err_phys": abs(e),
        })

    df_preds = pd.DataFrame(rows)
    metrics = dict(
        val_mae_phys=float(np.mean(abs_err_phys)) if abs_err_phys else float("nan"),
        val_rmse_phys=float(np.sqrt(np.mean(sq_err_phys))) if sq_err_phys else float("nan"),
        bias_phys=float(np.mean(err_phys)) if err_phys else float("nan"),
    )
    return metrics, df_preds


# ---------- public entrypoint ----------
def train_regression_low_tf(train_csv: str, val_csv: str, crop_box, cfg: dict):
    os.makedirs(cfg["runs_dir"], exist_ok=True)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)

    y_mean = float(train_df["log_axial_resolution"].mean())
    y_std  = float(train_df["log_axial_resolution"].std() + 1e-8)
    y_mean_t = torch.tensor(y_mean, device=device)
    y_std_t  = torch.tensor(y_std, device=device)

    # --- defaults for new cfg keys (safe if your cfg file doesn’t have them yet)
    sampling_mode = cfg.get("sampling_mode", "contiguous")
    uniform_eps_sec = cfg.get("uniform_eps_sec", 1.0/30.0)
    uniform_jitter = cfg.get("uniform_jitter", 0.2)

    train_ds = PrinterVideoDatasetX3DReg(
        train_df, crop_box=crop_box,
        clip_duration=cfg["clip_duration"],
        num_frames=cfg["num_frames"],
        image_size=cfg["image_size"],
        clips_per_video=cfg["clips_train"],
        random_start=True,
        seed=cfg["seed"],
        sampling_mode=sampling_mode,
        eps_sec=uniform_eps_sec,
        jitter=uniform_jitter,
    )
    val_ds = PrinterVideoDatasetX3DReg(
        val_df, crop_box=crop_box,
        clip_duration=cfg["clip_duration"],
        num_frames=cfg["num_frames"],
        image_size=cfg["image_size"],
        clips_per_video=cfg["clips_val"],
        random_start=False,  # deterministic eval
        seed=cfg["seed"],
        sampling_mode=sampling_mode,
        eps_sec=uniform_eps_sec,
        jitter=0.0,          # no jitter in eval
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"]
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"]
    )

    # not sure if i need this....
    # Make sure run name reflects sampling
    cfg = dict(cfg)  # avoid mutating caller
    cfg["sampling_mode"] = sampling_mode
    cfg["uniform_eps_sec"] = uniform_eps_sec
    cfg["uniform_jitter"] = uniform_jitter

    run_name = make_run_name(cfg)
    run_dir = os.path.join(cfg["runs_dir"], run_name)
    os.makedirs(run_dir, exist_ok=True)

    model = build_model_reg(cfg, device)
    criterion = nn.SmoothL1Loss(beta=float(cfg["huber_beta"]))
    early = EarlyStopping(patience=cfg["early_patience"], min_delta=cfg["min_delta"])

    best_val = float("inf")
    best_state = None
    best_epoch = None
    history = []
    total_epochs = 0

    # Phase A: head
    if cfg["freeze"]:
        freeze_backbone(model)
        opt = torch.optim.AdamW(model.head.parameters(), lr=cfg["lr_head"], weight_decay=cfg["weight_decay"])
        for e in range(cfg["epochs_head"]):
            total_epochs += 1
            tr = run_epoch_reg(model, train_loader, opt, criterion, device, True, y_mean_t, y_std_t)
            va = run_epoch_reg(model, val_loader, opt, criterion, device, False, y_mean_t, y_std_t)
            vmetrics, _ = video_level_eval_reg(model, val_loader, device, y_mean, y_std)
            history.append({
                "epoch": total_epochs,
                "phase": "head",
                "train_loss": float(tr),
                "val_loss": float(va),
                "val_mae_phys": float(vmetrics["val_mae_phys"]),
                "val_rmse_phys": float(vmetrics["val_rmse_phys"]),
                "bias_phys": float(vmetrics["bias_phys"]),
            })

            if cfg["print_every_epoch"]:
                print(f"[Head {e+1}/{cfg['epochs_head']}] tr {tr:.4f} va {va:.4f} | MAE {vmetrics['val_mae_phys']:.4f}")
            if va < best_val:
                best_val = va
                best_state = deepcopy(model.state_dict())
                best_epoch = total_epochs

    # Phase B: finetune
    unfreeze_all(model)
    opt = torch.optim.AdamW(
        [{"params": model.backbone.parameters(), "lr": cfg["lr_backbone"]},
         {"params": model.head.parameters(), "lr": cfg["lr_head"]}],
        weight_decay=cfg["weight_decay"]
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=cfg["sched_patience"]
    )

    for e in range(cfg["epochs_ft"]):
        total_epochs += 1
        tr = run_epoch_reg(model, train_loader, opt, criterion, device, True, y_mean_t, y_std_t)
        va = run_epoch_reg(model, val_loader, opt, criterion, device, False, y_mean_t, y_std_t)
        vmetrics, df_preds = video_level_eval_reg(model, val_loader, device, y_mean, y_std)
        history.append({
            "epoch": total_epochs,
            "phase": "ft",
            "train_loss": float(tr),
            "val_loss": float(va),
            "val_mae_phys": float(vmetrics["val_mae_phys"]),
            "val_rmse_phys": float(vmetrics["val_rmse_phys"]),
            "bias_phys": float(vmetrics["bias_phys"]),
        })

        sched.step(va)

        if cfg["print_every_epoch"]:
            print(f"[FT {e+1}/{cfg['epochs_ft']}] tr {tr:.4f} va {va:.4f} | MAE {vmetrics['val_mae_phys']:.4f}")

        if va < best_val:
            best_val = va
            best_state = deepcopy(model.state_dict())
            best_epoch = total_epochs

        if early.step(va, model):
            print(f"🛑 Early stop. best_val={early.best:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics, df_preds = video_level_eval_reg(
        model, val_loader, device, y_mean, y_std
    )

    # -------------------- save artifacts --------------------
    torch.save(model.state_dict(), os.path.join(run_dir, "model_best.pth"))
    df_preds.to_csv(os.path.join(run_dir, "val_video_predictions.csv"), index=False)

    # -------------------- learning curves --------------------
    hist_df = pd.DataFrame(history)
    hist_csv = os.path.join(run_dir, "learning_curves.csv")
    hist_df.to_csv(hist_csv, index=False)

    def plot_curve(x, y, title, ylabel, out_png):
        plt.figure(figsize=(8, 5))
        plt.plot(x, y)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

    if len(hist_df) > 0:
        plot_curve(
            hist_df["epoch"],
            hist_df["train_loss"],
            "Train Loss vs Epoch",
            "Train loss",
            os.path.join(run_dir, "curve_train_loss.png"),
        )
        plot_curve(
            hist_df["epoch"],
            hist_df["val_loss"],
            "Val Loss vs Epoch",
            "Val loss",
            os.path.join(run_dir, "curve_val_loss.png"),
        )
        plot_curve(
            hist_df["epoch"],
            hist_df["val_mae_phys"],
            "Val MAE (phys) vs Epoch",
            "MAE (phys units)",
            os.path.join(run_dir, "curve_val_mae_phys.png"),
        )
        plot_curve(
            hist_df["epoch"],
            hist_df["val_rmse_phys"],
            "Val RMSE (phys) vs Epoch",
            "RMSE (phys units)",
            os.path.join(run_dir, "curve_val_rmse_phys.png"),
        )

    print("✅ Saved learning curves:", hist_csv)
    print("✅ Saved curve plots in:", run_dir)

    # -------------------- run metadata --------------------
    payload = dict(
        cfg=cfg,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        best_epoch=best_epoch,
        best_val_loss=float(best_val),
        y_mean=y_mean,
        y_std=y_std,
        final_metrics=final_metrics,
        run_dir=run_dir,
        device=str(device),
    )
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # -------------------- append results row --------------------
    row = dict(
        timestamp=payload["timestamp"],
        model_name=cfg["model_name"],
        run_name=run_name,
        run_dir=run_dir,
        **{
            k: cfg[k]
            for k in [
                "freeze",
                "head_dropout",
                "lr_head",
                "lr_backbone",
                "weight_decay",
                "huber_beta",
                "clip_duration",
                "num_frames",
                "image_size",
                "clips_train",
                "clips_val",
                "batch_size",
                "seed",
                "sampling_mode",
                "uniform_eps_sec",
                "uniform_jitter",
            ]
            if k in cfg
        },
        **final_metrics,
    )
    def append_results_row(results_csv: str, row: dict):
        os.makedirs(os.path.dirname(results_csv), exist_ok=True)
        df_row = pd.DataFrame([row])

        if not os.path.exists(results_csv):
            df_row.to_csv(results_csv, index=False)
            return

        df_old = pd.read_csv(results_csv)
    
        # union of all columns, preserving existing order first
        all_cols = list(df_old.columns)
        for c in df_row.columns:
            if c not in all_cols:
                all_cols.append(c)
    
        df_old = df_old.reindex(columns=all_cols)
        df_row = df_row.reindex(columns=all_cols)
    
        df_new = pd.concat([df_old, df_row], ignore_index=True)
        df_new.to_csv(results_csv, index=False)

        return os.path.exists(results_csv)
        
    ok = append_results_row(cfg["results_csv"], row)
    if ok: 
        print("✅ Appended:", cfg["results_csv"])
    else: 
        print("❌ Failed to append:", cfg["results_csv"])

    print("✅ Saved:", run_dir)
    print("✅ Final metrics:", final_metrics)
    return run_dir, final_metrics

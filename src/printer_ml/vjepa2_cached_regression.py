import random

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import AutoModel


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# --------------------------------------------------
# Video loading
# --------------------------------------------------

def sample_video_frames(
    video_path: str,
    num_frames: int = 64,
    image_size: int = 256,
    crop_box=None,
):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {video_path}")

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
    ).astype(int)

    frames = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
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

            if x2 <= x1 or y2 <= y1:
                cap.release()
                raise ValueError(
                    f"Invalid crop_box={crop_box} for frame size width={w}, height={h}"
                )

            frame = frame[y1:y2, x1:x2]

        frame = cv2.resize(frame, (image_size, image_size))

        frame = frame.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        frame = (frame - mean) / std
        frame = np.transpose(frame, (2, 0, 1))

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No readable frames from video: {video_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1])

    frames = np.stack(frames, axis=0)

    return torch.tensor(frames, dtype=torch.float32)


# --------------------------------------------------
# Target transforms
# --------------------------------------------------

def compute_target_stats(
    df_train: pd.DataFrame,
    target_col: str,
    use_log_target: bool = True,
):
    y = df_train[target_col].astype(float).to_numpy()

    if use_log_target:
        if np.any(y <= 0):
            raise ValueError(
                "All target values must be positive when use_log_target=True"
            )
        y = np.log(y)

    target_mean = float(y.mean())
    target_std = float(y.std() + 1e-8)

    return target_mean, target_std


def transform_target(
    y,
    target_mean: float,
    target_std: float,
    use_log_target: bool = True,
):
    y = float(y)

    if use_log_target:
        if y <= 0:
            raise ValueError(
                f"Target must be positive when use_log_target=True. Got y={y}"
            )
        y = np.log(y)

    y = (y - target_mean) / target_std

    return float(y)


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
# Dataset for video embedding extraction
# --------------------------------------------------

class VideoEmbeddingDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        video_col: str,
        target_col: str,
        num_frames: int = 64,
        image_size: int = 256,
        target_mean: float = None,
        target_std: float = None,
        use_log_target: bool = True,
        crop_box=None,
    ):
        self.df = df.reset_index(drop=True)

        self.video_col = video_col
        self.target_col = target_col

        self.num_frames = num_frames
        self.image_size = image_size

        self.target_mean = target_mean
        self.target_std = target_std
        self.use_log_target = use_log_target

        self.crop_box = crop_box

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        video_path = row[self.video_col]
        target_phys = float(row[self.target_col])

        video = sample_video_frames(
            video_path=video_path,
            num_frames=self.num_frames,
            image_size=self.image_size,
            crop_box=self.crop_box,
        )

        target_norm = transform_target(
            target_phys,
            target_mean=self.target_mean,
            target_std=self.target_std,
            use_log_target=self.use_log_target,
        )

        video_id = row["video_id"] if "video_id" in row else str(idx)

        return {
            "video": video,
            "target_norm": torch.tensor(target_norm, dtype=torch.float32),
            "target_phys": torch.tensor(target_phys, dtype=torch.float32),
            "video_path": video_path,
            "video_id": video_id,
        }


# --------------------------------------------------
# V-JEPA2 embedder
# --------------------------------------------------

class VJEPA2Embedder(nn.Module):
    def __init__(
        self,
        hf_model_name: str = "facebook/vjepa2-vitl-fpc64-256",
    ):
        super().__init__()

        self.hf_model_name = hf_model_name

        self.backbone = AutoModel.from_pretrained(
            hf_model_name,
            attn_implementation="sdpa",
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.embed_dim = int(self.backbone.config.hidden_size)

    @torch.no_grad()
    def forward(self, video):
        outputs = self.backbone(
            pixel_values_videos=video,
            skip_predictor=True,
        )

        token_embeddings = outputs.last_hidden_state
        video_embedding = token_embeddings.mean(dim=1)

        return video_embedding


# --------------------------------------------------
# Dataset for cached embeddings
# --------------------------------------------------

class CachedEmbeddingDataset(Dataset):
    def __init__(self, embeddings_path: str):
        data = torch.load(
            embeddings_path,
            map_location="cpu",
            weights_only=False,
        )

        self.embeddings = data["embeddings"].float()
        self.targets = data["targets"].float()
        self.targets_phys = data["targets_phys"].float()

        self.video_paths = data["video_paths"]
        self.video_ids = data["video_ids"]

        self.target_mean = float(data["target_mean"])
        self.target_std = float(data["target_std"])
        self.use_log_target = bool(data["use_log_target"])

        self.df = data["df"]
        self.cfg = data["cfg"]

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]


# --------------------------------------------------
# Regression head
# --------------------------------------------------

class EmbeddingRegressor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings):
        prediction = self.regressor(embeddings)
        return prediction.squeeze(1)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate_cached_regression(
    model,
    loader,
    device,
    target_mean: float,
    target_std: float,
    use_log_target: bool = True,
):
    model.eval()

    preds_norm = []
    targets_norm = []

    with torch.no_grad():
        for embeddings, targets in loader:
            embeddings = embeddings.to(device)
            targets = targets.to(device)

            preds = model(embeddings)

            preds_norm.append(preds.detach().cpu().numpy())
            targets_norm.append(targets.detach().cpu().numpy())

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

    return {
        "val_mae_phys": float(mae_phys),
        "val_rmse_phys": float(rmse_phys),
        "bias_phys": float(bias_phys),
    }


def predict_cached_physical(
    model,
    loader,
    device,
    target_mean: float,
    target_std: float,
    use_log_target: bool = True,
):
    model.eval()

    preds_norm = []
    targets_norm = []

    with torch.no_grad():
        for embeddings, targets in loader:
            embeddings = embeddings.to(device)
            targets = targets.to(device)

            preds = model(embeddings)

            preds_norm.append(preds.detach().cpu().numpy())
            targets_norm.append(targets.detach().cpu().numpy())

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

    return preds_phys, targets_phys
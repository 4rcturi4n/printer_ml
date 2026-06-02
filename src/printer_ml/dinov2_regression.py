import random

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset

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
# Video loading
# --------------------------------------------------

def sample_video_frames(
    video_path: str,
    num_frames: int = 16,
    image_size: int = 224,
    crop_box=None,
):
    """
    Loads a video, uniformly samples frames, crops each frame if crop_box is given,
    then resizes to image_size x image_size.

    crop_box format:
        (x1, y1, x2, y2)

    Returns:
        Tensor of shape [T, 3, H, W]
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        num_frames
    ).astype(int)

    frames = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()

        if not success:
            continue

        # OpenCV gives BGR, DINOv2 expects RGB-like input
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --------------------------------------------------
        # Crop before resizing
        # --------------------------------------------------
        if crop_box is not None:
            x1, y1, x2, y2 = crop_box

            h, w = frame.shape[:2]

            x1 = max(0, min(int(x1), w))
            x2 = max(0, min(int(x2), w))
            y1 = max(0, min(int(y1), h))
            y2 = max(0, min(int(y2), h))

            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    f"Invalid crop_box={crop_box} for frame size width={w}, height={h}"
                )

            frame = frame[y1:y2, x1:x2]

        # Resize cropped frame to DINOv2 input size
        frame = cv2.resize(frame, (image_size, image_size))

        # Convert to [0, 1]
        frame = frame.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        frame = (frame - mean) / std

        # [H, W, C] -> [C, H, W]
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
# Dataset
# --------------------------------------------------

class VideoRegressionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        video_col: str,
        target_col: str,
        num_frames: int,
        image_size: int,
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
        y = float(row[self.target_col])

        frames = sample_video_frames(
            video_path=video_path,
            num_frames=self.num_frames,
            image_size=self.image_size,
            crop_box=self.crop_box,
        )

        if self.use_log_target:
            y = np.log(y)

        if self.target_mean is not None and self.target_std is not None:
            y = (y - self.target_mean) / self.target_std

        y = torch.tensor(y, dtype=torch.float32)

        return frames, y


# --------------------------------------------------
# Model
# --------------------------------------------------

def get_dinov2_embed_dim(model_name: str):
    if model_name == "dinov2_vits14":
        return 384
    elif model_name == "dinov2_vitb14":
        return 768
    elif model_name == "dinov2_vitl14":
        return 1024
    elif model_name == "dinov2_vitg14":
        return 1536
    else:
        raise ValueError(f"Unknown DINOv2 model: {model_name}")


class DINOv2VideoRegressor(nn.Module):
    def __init__(
        self,
        dinov2_name: str = "dinov2_vits14",
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.dinov2_name = dinov2_name

        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            dinov2_name
        )

        # Freeze DINOv2
        for param in self.backbone.parameters():
            param.requires_grad = False

        embed_dim = get_dinov2_embed_dim(dinov2_name)

        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, video):
        """
        Args:
            video: [B, T, 3, H, W]

        Returns:
            prediction: [B]
        """

        B, T, C, H, W = video.shape

        # [B, T, 3, H, W] -> [B*T, 3, H, W]
        frames = video.view(B * T, C, H, W)

        # Extract DINOv2 embeddings without training DINOv2
        with torch.no_grad():
            frame_embeddings = self.backbone(frames)

        # [B*T, D] -> [B, T, D]
        frame_embeddings = frame_embeddings.view(B, T, -1)

        # Average frame embeddings into one video embedding
        video_embedding = frame_embeddings.mean(dim=1)

        prediction = self.regressor(video_embedding)

        return prediction.squeeze(1)


# --------------------------------------------------
# Target inverse transform
# --------------------------------------------------

def inverse_transform_target(
    y_norm,
    target_mean: float,
    target_std: float,
    use_log_target: bool = True,
):
    """
    Converts normalized/log target back to physical axial resolution.
    """

    y = y_norm * target_std + target_mean

    if use_log_target:
        y = np.exp(y)

    return y


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate_regression(
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
        for videos, targets in loader:
            videos = videos.to(device)
            targets = targets.to(device)

            preds = model(videos)

            preds_norm.append(preds.cpu().numpy())
            targets_norm.append(targets.cpu().numpy())

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
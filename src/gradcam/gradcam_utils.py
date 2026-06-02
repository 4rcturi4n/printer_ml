import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def unnormalize_video(video, mean, std):
    """
    video: [C, T, H, W]
    """
    mean = torch.tensor(mean, dtype=video.dtype, device=video.device).view(-1, 1, 1, 1)
    std = torch.tensor(std, dtype=video.dtype, device=video.device).view(-1, 1, 1, 1)
    return video * std + mean


def _to_display_frame(frame_chw):
    frame = frame_chw.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    frame = frame - frame.min()
    frame = frame / (frame.max() + 1e-8)
    return frame


def save_gradcam_frames(video_tensor, cam_tensor, save_path, frame_indices=None, alpha=0.4, cmap="jet"):
    """
    video_tensor: [C, T, H, W]
    cam_tensor:   [T, H, W]
    Saves one PNG with:
      row 1 -> frame + Grad-CAM overlay
      row 2 -> original frame
    """
    if video_tensor.ndim != 4:
        raise ValueError(f"video_tensor must be [C,T,H,W], got {tuple(video_tensor.shape)}")
    if cam_tensor.ndim != 3:
        raise ValueError(f"cam_tensor must be [T,H,W], got {tuple(cam_tensor.shape)}")

    _, t, _, _ = video_tensor.shape
    t_cam, _, _ = cam_tensor.shape
    if t != t_cam:
        raise ValueError(f"time mismatch: video T={t}, cam T={t_cam}")

    if frame_indices is None:
        if t >= 4:
            frame_indices = [0, t // 3, 2 * t // 3, t - 1]
        else:
            frame_indices = list(range(t))

    frame_indices = [int(i) for i in frame_indices if 0 <= int(i) < t]
    if len(frame_indices) == 0:
        raise ValueError("No valid frame_indices")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    num_cols = len(frame_indices)
    plt.figure(figsize=(3 * num_cols, 6))

    for plot_idx, frame_idx in enumerate(frame_indices, start=1):
        frame = _to_display_frame(video_tensor[:, frame_idx])

        heatmap = cam_tensor[frame_idx].detach().cpu().numpy().astype(np.float32)
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)

        # Row 1: overlay
        plt.subplot(2, num_cols, plot_idx)
        plt.imshow(frame)
        plt.imshow(heatmap, cmap=cmap, alpha=alpha, vmin=0.0, vmax=1.0)
        plt.title(f"Frame {frame_idx} (CAM)")
        plt.axis("off")

        # Row 2: original
        plt.subplot(2, num_cols, plot_idx + num_cols)
        plt.imshow(frame)
        plt.title(f"Frame {frame_idx} (Original)")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
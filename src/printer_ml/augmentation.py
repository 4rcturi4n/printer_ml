"""
Frame-level video augmentation + variant planning + embedding-space augmentation.

Two separate stages live here, matching the pipeline discussed:

1. FRAME-LEVEL variants (used once, offline, before the frozen DINOv2 encoder).
   Each variant is a real, distinct image transform (blur / brightness / noise /
   crop-jitter) applied to the raw video frames before encoding. Expensive
   (one encoder forward pass per variant) so it's done once and cached.

2. EMBEDDING-SPACE augmentation (used every epoch, on the already-cached
   [T, D] tensors, no encoder involved). Cheap synthetic perturbation
   (noise / timestep masking / time-warp / mixup) used purely as a
   regularizer against overfitting on a small dataset.
"""

import random

import cv2
import numpy as np
import pandas as pd
import torch


# ====================================================
# 1. Frame-level variants (applied pre-encoder)
# ====================================================

def _variant_clean(frame: np.ndarray) -> np.ndarray:
    return frame


def _variant_blur(frame: np.ndarray) -> np.ndarray:
    k = random.choice([3, 5])
    return cv2.GaussianBlur(frame, (k, k), 0)


def _variant_brightness(frame: np.ndarray) -> np.ndarray:
    factor = random.uniform(0.75, 1.3)
    out = frame.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def _variant_noise(frame: np.ndarray) -> np.ndarray:
    sigma = random.uniform(5.0, 15.0)
    noise = np.random.normal(0.0, sigma, size=frame.shape).astype(np.float32)
    out = frame.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def _variant_crop_jitter(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    jitter_frac = 0.05
    dy = int(h * jitter_frac)
    dx = int(w * jitter_frac)
    y0 = random.randint(0, max(dy, 1))
    x0 = random.randint(0, max(dx, 1))
    y1 = h - random.randint(0, max(dy, 1))
    x1 = w - random.randint(0, max(dx, 1))
    cropped = frame[y0:y1, x0:x1]
    return cv2.resize(cropped, (w, h))


VARIANT_FUNCS = {
    "clean": _variant_clean,
    "blur": _variant_blur,
    "bright": _variant_brightness,
    "noise": _variant_noise,
    "crop": _variant_crop_jitter,
}


def sample_video_frames_variant(
    video_path: str,
    variant_name: str,
    num_frames: int = 32,
    image_size: int = 224,
    crop_box=None,
    skip_seconds: float = 2.0,
):
    """
    Same loading logic as mamba_early_prediction.sample_video_frames, but
    applies a named frame-level augmentation BEFORE resize/normalize.

    skip_seconds: skip this many seconds from the start of the video before
    sampling — no video in this dataset starts printing before this point,
    so it's a safe floor that trims dead time without risking real signal.
    """
    augment_fn = VARIANT_FUNCS[variant_name]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(fps * skip_seconds) if fps > 0 else 0
    skip_frames = min(skip_frames, max(total_frames - 1, 0))

    frame_indices = np.linspace(skip_frames, total_frames - 1, num_frames).astype(int)
    mean_arr = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std_arr = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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

        frame = augment_fn(frame)

        frame = cv2.resize(frame, (image_size, image_size))
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - mean_arr) / std_arr
        frame = np.transpose(frame, (2, 0, 1))
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No readable frames: {video_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1])

    return torch.tensor(np.stack(frames[:num_frames], axis=0), dtype=torch.float32)


def build_variant_plan(
    df: pd.DataFrame,
    target_col: str,
    n_bins: int = 4,
    min_variants: int = 1,
    max_variants: int = 5,
    global_bin_edges: np.ndarray = None,
) -> dict:
    """
    Decide how many (and which) augmented variants each video gets, based on
    how rare its target bin is. Rare-bin videos get more variants
    (real, distinct DINOv2-encoded copies); common-bin videos get just the
    clean version.

    global_bin_edges: bin edges computed from the FULL dataset using
        np.histogram or pd.qcut on log(target). If provided, these fixed
        edges are used instead of recomputing from df. This ensures that
        truly rare videos are always rare regardless of fold composition.

    Returns: {video_id: [variant_name, ...]}
    """
    log_vals = np.log(df[target_col].values.astype(float))

    if global_bin_edges is not None:
        # Use fixed global edges — rare videos stay rare across all folds
        bin_codes_arr = np.digitize(log_vals, global_bin_edges[1:-1])
    else:
        # Fallback: compute from this df (not recommended for fold subsets)
        _, edges = np.histogram(log_vals, bins=n_bins)
        bin_codes_arr = np.digitize(log_vals, edges[1:-1])

    bin_codes = pd.Series(bin_codes_arr, index=df.index)
    bin_counts = bin_codes.value_counts()

    max_count = bin_counts.max()
    min_count = bin_counts.min()

    augment_pool = ["blur", "bright", "noise", "crop"]
    plan = {}

    for video_id, code in zip(df["video_id"], bin_codes_arr):
        count = bin_counts.get(code, 1)
        if max_count == min_count:
            rarity = 0.0
        else:
            rarity = (max_count - count) / (max_count - min_count)

        n_variants = int(round(min_variants + rarity * (max_variants - min_variants)))
        n_variants = max(min_variants, min(max_variants, n_variants))

        variants = ["clean"] + augment_pool[: max(0, n_variants - 1)]
        plan[video_id] = variants

    return plan


# ====================================================
# 2. Embedding-space augmentation (applied every epoch, cheap)
# ====================================================

class EmbeddingAugmenter:
    """
    Applies cheap, synthetic perturbations directly to cached [T, D]
    embedding tensors at train time. No encoder involved.

    mixup_pool: optional list of (embedding [T,D], target_phys float) pairs
    to mix with. If None, mixup is skipped even if mixup_prob > 0.
    """

    def __init__(
        self,
        noise_prob: float = 0.8,
        noise_sigma: float = 0.02,
        mask_prob: float = 0.3,
        mask_frac: float = 0.1,
        warp_prob: float = 0.2,
        warp_strength: float = 0.3,
        mixup_prob: float = 0.2,
        mixup_alpha: float = 0.3,
        mixup_target_tol: float = None,
    ):
        self.noise_prob = noise_prob
        self.noise_sigma = noise_sigma
        self.mask_prob = mask_prob
        self.mask_frac = mask_frac
        self.warp_prob = warp_prob
        self.warp_strength = warp_strength
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        self.mixup_target_tol = mixup_target_tol

    def _add_noise(self, emb: torch.Tensor) -> torch.Tensor:
        std = emb.std(dim=0, keepdim=True).clamp_min(1e-6)
        return emb + torch.randn_like(emb) * self.noise_sigma * std

    def _mask_timesteps(self, emb: torch.Tensor) -> torch.Tensor:
        T = emb.shape[0]
        n_mask = max(1, int(T * self.mask_frac))
        idxs = random.sample(range(T), n_mask)
        out = emb.clone()
        for t in idxs:
            out[t] = out[t - 1] if t > 0 else out[t]
        return out

    def _time_warp(self, emb: torch.Tensor) -> torch.Tensor:
        T = emb.shape[0]
        steps = np.random.uniform(
            1.0 - self.warp_strength, 1.0 + self.warp_strength, size=T
        )
        warp = np.cumsum(steps)
        warp = (warp - warp[0]) / (warp[-1] - warp[0]) * (T - 1)
        orig_idx = np.arange(T)

        emb_np = emb.numpy()
        warped = np.stack(
            [np.interp(warp, orig_idx, emb_np[:, d]) for d in range(emb_np.shape[1])],
            axis=1,
        )
        return torch.tensor(warped, dtype=emb.dtype)

    def _mixup(self, emb, target_phys, pool):
        candidates = pool
        if self.mixup_target_tol is not None:
            candidates = [
                (e, y) for (e, y) in pool if abs(y - target_phys) <= self.mixup_target_tol
            ]
        if not candidates:
            return emb, target_phys

        other_emb, other_y = random.choice(candidates)
        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        mixed_emb = lam * emb + (1 - lam) * other_emb
        mixed_y = lam * target_phys + (1 - lam) * other_y
        return mixed_emb, mixed_y

    def __call__(self, emb: torch.Tensor, target_phys: float, mixup_pool=None):
        if random.random() < self.noise_prob:
            emb = self._add_noise(emb)

        choice = random.random()
        if choice < self.mask_prob:
            emb = self._mask_timesteps(emb)
        elif choice < self.mask_prob + self.warp_prob:
            emb = self._time_warp(emb)
        elif mixup_pool and choice < self.mask_prob + self.warp_prob + self.mixup_prob:
            emb, target_phys = self._mixup(emb, target_phys, mixup_pool)

        return emb, target_phys

"""
Extract frozen-DINOv2 embeddings for every video, once, and cache them to disk.

This is the offline step ("step 3" in the augmentation plan): for each
video we generate 1+ frame-level variants (clean + augmented copies),
run each one through DINOv2 once, and save the resulting [T, D] embedding
sequences. Rare-target-bin videos get more variants than common ones.

What this produces
-------------------
results/dinov2_embeddings/
  config.json
  fold_1/
    embedding_summary.json
    train_embeddings.pt   <- one row per (video, variant); includes
                              'video_id', 'variant_name', 'is_augmented'
    val_embeddings.pt     <- one row per video (clean variant only)

Train your cached Mamba head on train_embeddings.pt with
scripts/train_mamba_cached_kfold.py.
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC = os.path.join(ROOT, "src")

sys.path.insert(0, SRC)
sys.path.insert(0, ROOT)

from configs.project import CROP_BOX
from printer_ml.mamba_early_prediction import set_seed
from printer_ml.augmentation import build_variant_plan, sample_video_frames_variant


# --------------------------------------------------
# Config
# --------------------------------------------------

CFG = {
    "split_dir": os.path.join(ROOT, "data", "processed", "kfold_splits"),
    "out_dir":   os.path.join(ROOT, "results", "dinov2_embeddings"),

    "video_col":  "video_path",
    "target_col": "axial_resolution",

    "n_splits": 5,

    "dinov2_model": "dinov2_vits14",
    "num_frames":   32,
    "image_size":   224,

    "crop_box": CROP_BOX,

    "use_log_target": True,

    # variant planning — how many augmented copies rare-target videos get
    "n_bins":        4,
    "min_variants":  1,
    "max_variants":  5,

    "seed": 42,
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def get_fold_paths(split_dir, fold):
    train_csv = os.path.join(split_dir, f"fold_{fold}_train.csv")
    val_csv = os.path.join(split_dir, f"fold_{fold}_val.csv")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train split not found: {train_csv}")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Validation split not found: {val_csv}")

    return train_csv, val_csv


def compute_target_stats(df_train, target_col, use_log_target):
    y = df_train[target_col].astype(float).to_numpy()
    if use_log_target:
        y = np.log(y)
    return float(y.mean()), float(y.std() + 1e-8)


def transform_target(y, target_mean, target_std, use_log_target):
    y = float(y)
    if use_log_target:
        y = np.log(y)
    return (y - target_mean) / target_std


@torch.no_grad()
def encode_frames(backbone, frames, device):
    """frames: [T, 3, H, W] -> [T, D]"""
    frames = frames.to(device)
    emb = backbone(frames)
    return emb.cpu()


def extract_split_embeddings(
    split_name,
    df,
    backbone,
    cfg,
    target_mean,
    target_std,
    device,
    out_path,
    variant_plan=None,
):
    """
    variant_plan: {video_id: [variant_name, ...]} or None (clean only, for val)
    """
    all_embeddings = []
    all_targets = []
    all_targets_phys = []
    all_video_ids = []
    all_variant_names = []
    all_is_augmented = []

    n_rows_planned = (
        sum(len(v) for v in variant_plan.values()) if variant_plan else len(df)
    )
    print(f"Extracting {split_name} embeddings: {len(df)} videos -> {n_rows_planned} rows")

    row_count = 0
    for _, row in df.iterrows():
        video_path = row[cfg["video_col"]]
        video_id = row["video_id"]
        target_phys = float(row[cfg["target_col"]])

        variants = variant_plan[video_id] if variant_plan else ["clean"]

        for variant_name in variants:
            frames = sample_video_frames_variant(
                video_path=video_path,
                variant_name=variant_name,
                num_frames=cfg["num_frames"],
                image_size=cfg["image_size"],
                crop_box=cfg["crop_box"],
            )

            emb = encode_frames(backbone, frames, device)  # [T, D]
            target_norm = transform_target(
                target_phys, target_mean, target_std, cfg["use_log_target"]
            )

            all_embeddings.append(emb)
            all_targets.append(torch.tensor(target_norm, dtype=torch.float32))
            all_targets_phys.append(torch.tensor(target_phys, dtype=torch.float32))
            all_video_ids.append(video_id)
            all_variant_names.append(variant_name)
            all_is_augmented.append(variant_name != "clean")

            row_count += 1
            print(f"{split_name} | {row_count}/{n_rows_planned} | video={video_id} variant={variant_name}")

    payload = {
        "embeddings": torch.stack(all_embeddings, dim=0),     # [N, T, D]
        "targets": torch.stack(all_targets, dim=0),           # [N]
        "targets_phys": torch.stack(all_targets_phys, dim=0), # [N]
        "video_ids": all_video_ids,
        "variant_names": all_variant_names,
        "is_augmented": all_is_augmented,
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "use_log_target": bool(cfg["use_log_target"]),
        "cfg": cfg,
        "split_name": split_name,
    }

    torch.save(payload, out_path)
    print(f"Saved {split_name} embeddings to: {out_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def extract_kfold_embeddings(cfg):
    set_seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    save_json(cfg, os.path.join(cfg["out_dir"], "config.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("DINOv2 K-fold embedding extraction (with augmented variants)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Split dir: {cfg['split_dir']}")
    print(f"Output dir: {cfg['out_dir']}")
    print("=" * 80)

    backbone = torch.hub.load("facebookresearch/dinov2", cfg["dinov2_model"]).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    summary = []

    for fold in range(1, cfg["n_splits"] + 1):
        print()
        print("=" * 80)
        print(f"Fold {fold}/{cfg['n_splits']}")
        print("=" * 80)

        fold_dir = os.path.join(cfg["out_dir"], f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_csv, val_csv = get_fold_paths(cfg["split_dir"], fold)
        df_train = pd.read_csv(train_csv).reset_index(drop=True)
        df_val = pd.read_csv(val_csv).reset_index(drop=True)

        target_mean, target_std = compute_target_stats(
            df_train, cfg["target_col"], cfg["use_log_target"]
        )

        # only the TRAIN split gets extra variants for rare bins;
        # val stays clean/untouched so evaluation is unbiased
        variant_plan = build_variant_plan(
            df_train,
            target_col=cfg["target_col"],
            n_bins=cfg["n_bins"],
            min_variants=cfg["min_variants"],
            max_variants=cfg["max_variants"],
        )

        fold_info = {
            "fold": fold,
            "train_csv": train_csv,
            "val_csv": val_csv,
            "num_train_videos": len(df_train),
            "num_train_rows": sum(len(v) for v in variant_plan.values()),
            "num_val": len(df_val),
            "target_mean": float(target_mean),
            "target_std": float(target_std),
            "train_embeddings": os.path.join(fold_dir, "train_embeddings.pt"),
            "val_embeddings": os.path.join(fold_dir, "val_embeddings.pt"),
        }
        save_json(fold_info, os.path.join(fold_dir, "embedding_summary.json"))
        save_json(variant_plan, os.path.join(fold_dir, "variant_plan.json"))

        extract_split_embeddings(
            split_name="train",
            df=df_train,
            backbone=backbone,
            cfg=cfg,
            target_mean=target_mean,
            target_std=target_std,
            device=device,
            out_path=fold_info["train_embeddings"],
            variant_plan=variant_plan,
        )

        extract_split_embeddings(
            split_name="val",
            df=df_val,
            backbone=backbone,
            cfg=cfg,
            target_mean=target_mean,
            target_std=target_std,
            device=device,
            out_path=fold_info["val_embeddings"],
            variant_plan=None,
        )

        summary.append(fold_info)

    save_json(
        {"cfg": cfg, "folds": summary},
        os.path.join(cfg["out_dir"], "embedding_summary_all_folds.json"),
    )

    print()
    print("=" * 80)
    print("DINOv2 embedding extraction finished")
    print("=" * 80)


if __name__ == "__main__":
    extract_kfold_embeddings(CFG)

import os
import sys
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader


# --------------------------------------------------
# Make src import work
# --------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC = os.path.join(ROOT, "src")

sys.path.insert(0, SRC)
sys.path.insert(0, ROOT)


from configs.project import CROP_BOX

from printer_ml.vjepa2_cached_regression import (
    set_seed,
    compute_target_stats,
    VideoEmbeddingDataset,
    VJEPA2Embedder,
)


# --------------------------------------------------
# Config
# --------------------------------------------------

CFG = {
    "split_dir": os.path.join(ROOT, "data", "processed", "kfold_splits"),
    "out_dir": os.path.join(ROOT, "results", "vjepa2_embeddings"),

    "video_col": "video_path",
    "target_col": "axial_resolution",

    "n_splits": 5,

    "hf_model_name": "facebook/vjepa2-vitl-fpc64-256",

    "num_frames": 64,
    "image_size": 256,

    "crop_box": CROP_BOX,

    "use_log_target": True,

    "batch_size": 1,
    "num_workers": 2,

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


@torch.no_grad()
def extract_split_embeddings(
    split_name,
    df,
    model,
    cfg,
    target_mean,
    target_std,
    device,
    out_path,
):
    dataset = VideoEmbeddingDataset(
        df=df,
        video_col=cfg["video_col"],
        target_col=cfg["target_col"],
        num_frames=cfg["num_frames"],
        image_size=cfg["image_size"],
        target_mean=target_mean,
        target_std=target_std,
        use_log_target=cfg["use_log_target"],
        crop_box=cfg["crop_box"],
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    model.eval()

    all_embeddings = []
    all_targets = []
    all_targets_phys = []
    all_video_paths = []
    all_video_ids = []

    print(f"Extracting {split_name} embeddings: {len(dataset)} videos")

    for step, batch in enumerate(loader, start=1):
        videos = batch["video"].to(device)

        embeddings = model(videos)

        all_embeddings.append(embeddings.cpu())
        all_targets.append(batch["target_norm"].cpu())
        all_targets_phys.append(batch["target_phys"].cpu())

        all_video_paths.extend(batch["video_path"])
        all_video_ids.extend(batch["video_id"])

        print(
            f"{split_name} | "
            f"{step}/{len(loader)} | "
            f"batch embeddings {tuple(embeddings.shape)}"
        )

    payload = {
        "embeddings": torch.cat(all_embeddings, dim=0),
        "targets": torch.cat(all_targets, dim=0),
        "targets_phys": torch.cat(all_targets_phys, dim=0),
        "video_paths": list(all_video_paths),
        "video_ids": list(all_video_ids),
        "df": df.reset_index(drop=True),
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "use_log_target": bool(cfg["use_log_target"]),
        "cfg": cfg,
        "split_name": split_name,
        "embed_dim": int(model.embed_dim),
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
    print("V-JEPA2 K-fold embedding extraction")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Split dir: {cfg['split_dir']}")
    print(f"Output dir: {cfg['out_dir']}")
    print(f"Crop box: {cfg['crop_box']}")
    print("=" * 80)

    model = VJEPA2Embedder(
        hf_model_name=cfg["hf_model_name"],
    ).to(device)

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
            df_train=df_train,
            target_col=cfg["target_col"],
            use_log_target=cfg["use_log_target"],
        )

        fold_info = {
            "fold": fold,
            "train_csv": train_csv,
            "val_csv": val_csv,
            "num_train": len(df_train),
            "num_val": len(df_val),
            "target_mean": float(target_mean),
            "target_std": float(target_std),
            "train_embeddings": os.path.join(fold_dir, "train_embeddings.pt"),
            "val_embeddings": os.path.join(fold_dir, "val_embeddings.pt"),
        }

        save_json(
            fold_info,
            os.path.join(fold_dir, "embedding_summary.json"),
        )

        extract_split_embeddings(
            split_name="train",
            df=df_train,
            model=model,
            cfg=cfg,
            target_mean=target_mean,
            target_std=target_std,
            device=device,
            out_path=fold_info["train_embeddings"],
        )

        extract_split_embeddings(
            split_name="val",
            df=df_val,
            model=model,
            cfg=cfg,
            target_mean=target_mean,
            target_std=target_std,
            device=device,
            out_path=fold_info["val_embeddings"],
        )

        summary.append(fold_info)

    summary_path = os.path.join(cfg["out_dir"], "embedding_summary_all_folds.json")
    save_json(
        {
            "cfg": cfg,
            "folds": summary,
        },
        summary_path,
    )

    print()
    print("=" * 80)
    print("V-JEPA2 embedding extraction finished")
    print("=" * 80)
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    extract_kfold_embeddings(CFG)
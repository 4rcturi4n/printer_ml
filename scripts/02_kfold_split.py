import sys
import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from configs.reg_base import CFG
from configs.project import CROP_BOX, VIDEO_XL_DATASET_CSV

from printer_ml.kfold_split import make_stratified_kfold_splits
from printer_ml.train_reg_low_tf import (
    train_regression_low_tf,
    PrinterVideoDatasetX3DReg,
    build_model_reg,
    make_run_name,
)

from gradcam.gradcam_video import VideoGradCAM
from gradcam.gradcam_utils import unnormalize_video, save_gradcam_frames


# --------------------------------------------------
# Load existing folds or create them once
# --------------------------------------------------

def load_or_create_kfold_splits(
    in_csv,
    out_dir,
    n_splits=5,
    n_bins=4,
    seed=42,
):
    """
    Load existing k-fold split CSVs if they exist.
    Otherwise create them once.

    This prevents accidentally changing folds between X3D and DINOv2.
    """

    fold_infos = []
    all_exist = True

    for fold in range(1, n_splits + 1):
        train_csv = os.path.join(out_dir, f"fold_{fold}_train.csv")
        val_csv = os.path.join(out_dir, f"fold_{fold}_val.csv")
        split_json = os.path.join(out_dir, f"fold_{fold}_split.json")

        if not os.path.exists(train_csv) or not os.path.exists(val_csv):
            all_exist = False
            break

        fold_infos.append(
            {
                "fold": fold,
                "train_csv": train_csv,
                "val_csv": val_csv,
                "split_json": split_json,
            }
        )

    if all_exist:
        print("✅ Using existing saved k-fold splits:")
        for info in fold_infos:
            print(f"   Fold {info['fold']}")
            print(f"      train: {info['train_csv']}")
            print(f"      val:   {info['val_csv']}")
            print(f"      meta:  {info['split_json']}")
        return fold_infos

    print("⚠️ Saved k-fold splits not found.")
    print("Creating k-fold splits now...")

    fold_infos = make_stratified_kfold_splits(
        in_csv=in_csv,
        out_dir=out_dir,
        n_splits=n_splits,
        n_bins=n_bins,
        seed=seed,
    )

    return fold_infos


# --------------------------------------------------
# Grad-CAM helpers
# --------------------------------------------------

def choose_target_layer(model):
    return model.backbone.blocks[4]


def run_gradcam_for_fold(run_dir, train_csv, val_csv, crop_box, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------
    # IMPORTANT:
    # Use the SAME train CSV that was used for this fold.
    # This gives the same y_mean and y_std as training.
    # -------------------------------------------------

    train_df_for_norm = pd.read_csv(train_csv)

    y_mean = float(train_df_for_norm["log_axial_resolution"].mean())
    y_std = float(train_df_for_norm["log_axial_resolution"].std() + 1e-8)

    print("Grad-CAM target normalization:")
    print(f"  train_csv: {train_csv}")
    print(f"  y_mean: {y_mean:.6f}")
    print(f"  y_std:  {y_std:.6f}")

    cfg = dict(cfg)
    sampling_mode = cfg.get("sampling_mode", "contiguous")
    uniform_eps_sec = cfg.get("uniform_eps_sec", 1.0 / 30.0)
    uniform_jitter = cfg.get("uniform_jitter", 0.2)

    ckpt_path = os.path.join(run_dir, "model_best.pth")
    out_dir = os.path.join(run_dir, "gradcam")
    os.makedirs(out_dir, exist_ok=True)

    print("Using run_dir:", run_dir)
    print("Loading checkpoint:", ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ----------------------------
    # Build model and load weights
    # ----------------------------

    model = build_model_reg(cfg, device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # ----------------------------
    # Build validation dataset
    # ----------------------------

    val_df = pd.read_csv(val_csv)

    val_ds = PrinterVideoDatasetX3DReg(
        val_df,
        crop_box=crop_box,
        clip_duration=cfg["clip_duration"],
        num_frames=cfg["num_frames"],
        image_size=cfg["image_size"],
        clips_per_video=cfg["clips_val"],
        random_start=False,
        seed=cfg["seed"],
        sampling_mode=sampling_mode,
        eps_sec=uniform_eps_sec,
        jitter=0.0,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    # ----------------------------
    # Grad-CAM
    # ----------------------------

    target_layer = choose_target_layer(model)
    print("Target layer:", target_layer)

    gradcam = VideoGradCAM(model, target_layer)

    batch = next(iter(val_loader))
    videos, targets, paths = batch

    videos = videos.to(device)
    targets = targets.to(device).float()

    with torch.enable_grad():
        cams, preds = gradcam.generate(videos)

    print("videos shape:", tuple(videos.shape))
    print("cams shape:", tuple(cams.shape))
    print("preds shape:", tuple(preds.shape))

    # ----------------------------
    # Save a few samples
    # ----------------------------

    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]

    num_to_save = min(4, videos.shape[0])

    for i in range(num_to_save):
        video_vis = unnormalize_video(
            videos[i].detach().cpu(),
            mean=mean,
            std=std,
        )

        num_frames = video_vis.shape[1]
        frame_indices = np.linspace(0, num_frames - 1, 8).astype(int).tolist()

        # -------------------------------------------------
        # IMPORTANT:
        # preds[i] is pred_norm, NOT pred_log.
        # targets[i] is true_log because your dataset returns raw log_axial_resolution.
        # -------------------------------------------------

        pred_norm = float(preds[i].detach().cpu().view(-1)[0])
        true_log = float(targets[i].detach().cpu().view(-1)[0])

        pred_log = pred_norm * y_std + y_mean

        pred_phys = float(np.exp(pred_log))
        true_phys = float(np.exp(true_log))

        safe_name = os.path.splitext(os.path.basename(paths[i]))[0]
        out_path = os.path.join(out_dir, f"{i:02d}_{safe_name}_gradcam.png")

        save_gradcam_frames(
            video_tensor=video_vis,
            cam_tensor=cams[i].detach().cpu(),
            save_path=out_path,
            frame_indices=frame_indices,
            alpha=0.4,
        )

        print(f"Saved: {out_path}")
        print(
            f"  true_log={true_log:.4f}  "
            f"pred_norm={pred_norm:.4f}  "
            f"pred_log={pred_log:.4f}"
        )
        print(
            f"  true_phys={true_phys:.4f}  "
            f"pred_phys={pred_phys:.4f}"
        )

    meta = {
        "run_dir": run_dir,
        "checkpoint": ckpt_path,
        "gradcam_dir": out_dir,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "y_mean": y_mean,
        "y_std": y_std,
        "target_layer": str(target_layer),
        "num_saved": int(num_to_save),
    }

    with open(os.path.join(out_dir, "gradcam_run.json"), "w") as f:
        json.dump(meta, f, indent=2)

    gradcam.remove_hooks()

    print("✅ Grad-CAM done.")


# --------------------------------------------------
# Learning curve summary
# --------------------------------------------------

def save_all_fold_learning_curves(fold_learning_curve_csvs, save_path):
    """
    Save one JPG containing all k-fold learning curves.

    The JPG contains:
    - train loss
    - val loss
    - val MAE
    - val RMSE
    """

    metrics = [
        ("train_loss", "Train Loss"),
        ("val_loss", "Validation Loss"),
        ("val_mae_phys", "Validation MAE Phys"),
        ("val_rmse_phys", "Validation RMSE Phys"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for fold_idx, csv_path in enumerate(fold_learning_curve_csvs, start=1):
        if not os.path.exists(csv_path):
            print(f"⚠️ Missing learning curve CSV, skipping: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        print(f"📈 Reading Fold {fold_idx} curves:")
        print(f"   {csv_path}")
        print(f"   columns: {list(df.columns)}")

        if "epoch" in df.columns:
            epochs = df["epoch"].values
        else:
            epochs = range(1, len(df) + 1)

        for ax, (metric_key, title) in zip(axes, metrics):
            if metric_key not in df.columns:
                print(f"⚠️ Column '{metric_key}' not found in {csv_path}")
                continue

            ax.plot(
                epochs,
                df[metric_key].values,
                label=f"Fold {fold_idx}",
            )

            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_key)
            ax.grid(True)

    for ax in axes:
        ax.legend()

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved combined k-fold learning curves: {save_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    cfg = dict(CFG)

    kfold_root = os.path.join("runs", "kfold_x3d")
    os.makedirs(kfold_root, exist_ok=True)

    split_dir = os.path.join("data", "processed", "kfold_splits")

    fold_infos = load_or_create_kfold_splits(
        in_csv=VIDEO_XL_DATASET_CSV,
        out_dir=split_dir,
        n_splits=5,
        n_bins=4,
        seed=cfg["seed"],
    )

    fold_learning_curve_csvs = []

    for info in fold_infos:
        fold = info["fold"]

        print("\n" + "=" * 80)
        print(f"🚀 TRAINING FOLD {fold}/5")
        print("=" * 80)

        cfg_fold = dict(cfg)

        fold_dir = os.path.join(kfold_root, f"fold_{fold}")

        cfg_fold["runs_dir"] = fold_dir

        # One shared results file for the whole k-fold run.
        # All folds append their final metrics here.
        cfg_fold["results_csv"] = os.path.join(kfold_root, "kfold_results.csv")

        # Useful if your training function stores fold info.
        cfg_fold["fold"] = fold

        run_dir, metrics = train_regression_low_tf(
            train_csv=info["train_csv"],
            val_csv=info["val_csv"],
            crop_box=CROP_BOX,
            cfg=cfg_fold,
        )

        print(f"✅ Fold {fold} metrics:", metrics)

        learning_curve_csv = os.path.join(run_dir, "learning_curves.csv")
        fold_learning_curve_csvs.append(learning_curve_csv)

        run_gradcam_for_fold(
            run_dir=run_dir,
            train_csv=info["train_csv"],
            val_csv=info["val_csv"],
            crop_box=CROP_BOX,
            cfg=cfg_fold,
        )

    save_all_fold_learning_curves(
        fold_learning_curve_csvs=fold_learning_curve_csvs,
        save_path=os.path.join(kfold_root, "all_folds_learning_curves.jpg"),
    )


if __name__ == "__main__":
    main()
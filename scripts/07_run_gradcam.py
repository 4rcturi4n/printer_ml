import sys
import os
import json
import torch
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from configs.reg_base import CFG
from configs.project import CROP_BOX, TRAIN_CSV, VAL_CSV

from printer_ml.train_reg_low_tf import (
    PrinterVideoDatasetX3DReg,
    build_model_reg,
    make_run_name,
)

from gradcam.gradcam_video import VideoGradCAM
from gradcam.gradcam_utils import unnormalize_video, save_gradcam_frames


def choose_target_layer(model):
    return model.backbone.blocks[4]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = dict(CFG)
    sampling_mode = cfg.get("sampling_mode", "contiguous")
    uniform_eps_sec = cfg.get("uniform_eps_sec", 1.0 / 30.0)
    uniform_jitter = cfg.get("uniform_jitter", 0.2)

    run_name = make_run_name({
        **cfg,
        "sampling_mode": sampling_mode,
    })
    run_dir = os.path.join(cfg["runs_dir"], run_name)

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
    val_df = pd.read_csv(VAL_CSV)

    val_ds = PrinterVideoDatasetX3DReg(
        val_df,
        crop_box=CROP_BOX,
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
        video_vis = unnormalize_video(videos[i].detach().cpu(), mean=mean, std=std)

        num_frames = video_vis.shape[1]
        frame_indices = np.linspace(0, num_frames - 1, 8).astype(int).tolist()
        print("num_frames:", num_frames)
        print("frame_indices:", frame_indices)

        pred_log = float(preds[i].detach().cpu().view(-1)[0])
        true_log = float(targets[i].detach().cpu().view(-1)[0])

        pred_phys = float(torch.exp(torch.tensor(pred_log)).item())
        true_phys = float(torch.exp(torch.tensor(true_log)).item())

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
        print(f"  true_log={true_log:.4f}  pred_log={pred_log:.4f}")
        print(f"  true_phys={true_phys:.4f}  pred_phys={pred_phys:.4f}")

    meta = {
        "run_dir": run_dir,
        "checkpoint": ckpt_path,
        "gradcam_dir": out_dir,
        "target_layer": str(target_layer),
        "num_saved": int(num_to_save),
    }
    with open(os.path.join(out_dir, "gradcam_run.json"), "w") as f:
        json.dump(meta, f, indent=2)

    gradcam.remove_hooks()
    print("Done.")


if __name__ == "__main__":
    main()
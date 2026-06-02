import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from printer_ml.reg_baselines import (
    RegConfig,
    get_3_reg_models,
    to_log,
    to_phys,
    fit_eval_one_model,
    compute_learning_curves,
    save_scatter_true_pred,
    save_residual_hist,
)

# ----------------------------
# Paths (EDIT)
# ----------------------------
X_NPY = "data/processed/features/X_meanstd.npy"                  # (N, D)
LABELS_CSV = None     # must contain axial_resolution
# If you instead have y_phys.npy, set LABELS_CSV=None and set Y_PHYS_NPY below.
Y_LOG_NPY = "data/processed/features/y_log_axial.npy"                        # e.g. "data/processed/y_phys.npy"

SAVE_DIR = "baseline_reg_ml"
RESULTS_CSV = os.path.join(SAVE_DIR, "results_reg_ml.csv")
SPLIT_DIR = os.path.join(SAVE_DIR, "split")     # will store train_idx.npy/val_idx.npy

CFG = RegConfig(
    save_dir=SAVE_DIR,
    results_csv=RESULTS_CSV,
    seed=0,
    y_phys_col="axial_resolution",
    y_log_col=None,
    split_col=None,          # none because you don't have it
    group_col=None,          # if you ever have group ids, we can add GroupShuffleSplit
    log_base="e",            # match your NN (e or 10)
)

VAL_FRAC = 0.2          # NN-like val split
N_BINS = 10             # stratify bins on y_log (will auto-reduce if dataset small)

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)


def load_y_log(n: int) -> np.ndarray:
    y_log = np.load(Y_LOG_NPY).astype(float).reshape(-1)
    if len(y_log) != n:
        raise ValueError(f"y_log length {len(y_log)} != X rows {n}")
    return y_log

def make_and_save_split(y_log: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(y_log)
    idx = np.arange(n)

    # adapt bins if dataset small
    n_bins = min(N_BINS, max(2, int(n / 20)))
    bins = pd.qcut(y_log, q=n_bins, duplicates="drop")

    train_idx, val_idx = train_test_split(
        idx,
        test_size=VAL_FRAC,
        random_state=CFG.seed,
        shuffle=True,
        stratify=bins
    )

    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)

    np.save(os.path.join(SPLIT_DIR, "train_idx.npy"), train_idx)
    np.save(os.path.join(SPLIT_DIR, "val_idx.npy"), val_idx)

    # also save a tiny summary for sanity
    summary = {
        "seed": CFG.seed,
        "val_frac": VAL_FRAC,
        "n": int(n),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "bins": int(n_bins),
    }
    with open(os.path.join(SPLIT_DIR, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return train_idx, val_idx


def main():
    # Load X
    X = np.load(X_NPY)
    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D (N,D), got {X.shape}")
    X = X.astype(float)
    n = X.shape[0]

    # Load y (log)
    y_log = load_y_log(n)

    # Create split (since you forgot)
    train_idx, val_idx = make_and_save_split(y_log)

    X_train, X_val = X[train_idx], X[val_idx]
    y_train_log, y_val_log = y_log[train_idx], y_log[val_idx]

    # Models
    models = get_3_reg_models(seed=CFG.seed)

    rows = []
    for name, model in models.items():
        print(f"\n=== {name} ===")

        row, pred_val_log = fit_eval_one_model(
            name, model,
            X_train, y_train_log,
            X_val, y_val_log,
            CFG
        )

        # save preds
        preds_csv = os.path.join(SAVE_DIR, f"preds_{name}.csv")
        pd.DataFrame({
            "idx": val_idx,
            "true_log": y_val_log,
            "pred_log": pred_val_log,
            "true_phys": to_phys(y_val_log, base=CFG.log_base),
            "pred_phys": to_phys(pred_val_log, base=CFG.log_base),
        }).to_csv(preds_csv, index=False)
        row["preds_csv"] = preds_csv

        # plots phys
        true_phys = to_phys(y_val_log, base=CFG.log_base)
        pred_phys = to_phys(pred_val_log, base=CFG.log_base)

        scatter_png = os.path.join(SAVE_DIR, f"scatter_{name}_phys.png")
        save_scatter_true_pred(true_phys, pred_phys, scatter_png, title=f"{name} (VAL) true vs pred (phys)")
        row["scatter_phys_png"] = scatter_png

        resid_png = os.path.join(SAVE_DIR, f"residuals_{name}_phys.png")
        save_residual_hist(pred_phys - true_phys, resid_png, title=f"{name} (VAL) residuals (phys)")
        row["residuals_phys_png"] = resid_png

        # learning curves
        lc_paths = compute_learning_curves(name, model, X_train, y_train_log, CFG)
        row.update(lc_paths)

        rows.append(row)

        print(
            f"VAL: mae_phys={row['val_mae_phys']:.4f} | rmse_phys={row['val_rmse_phys']:.4f} | "
            f"mae_log={row['val_mae_log']:.4f}"
        )

    out_df = pd.DataFrame(rows).sort_values("val_mae_phys", ascending=True)
    out_df.to_csv(RESULTS_CSV, index=False)
    print("\n✅ Saved results to:", RESULTS_CSV)
    print("✅ Saved split idx to:", SPLIT_DIR)

    # Save config snapshot
    with open(os.path.join(SAVE_DIR, "run_config.json"), "w") as f:
        json.dump(CFG.__dict__, f, indent=2)

    print("\nTop models by VAL MAE (phys):")
    print(out_df[["model", "val_mae_phys", "val_rmse_phys", "val_bias_phys", "val_mae_log"]].to_string(index=False))


if __name__ == "__main__":
    main()
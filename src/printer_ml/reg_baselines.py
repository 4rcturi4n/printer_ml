import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
# ----------------------------
# Config dataclass
# ----------------------------
@dataclass
class RegConfig:
    save_dir: str = "baseline_reg_ml"
    results_csv: str = "results_reg_ml.csv"
    seed: int = 0

    # where labels live
    y_phys_col: str = "axial_resolution"   # physical units column in your dataframe
    y_log_col: Optional[str] = None        # if you already have log column, set it here (e.g. "axial_log")

    # if you already have NN split column in your table: "train"/"val"/"test"
    split_col: Optional[str] = "split"

    # optional group split (if your NN split is group-based); only used if split_col missing
    group_col: Optional[str] = None

    # log base control
    log_base: str = "e"  # "e" or "10"

    # learning curve
    lc_train_sizes: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)


# ----------------------------
# Log transform helpers
# ----------------------------
def to_log(y_phys: np.ndarray, base: str = "e") -> np.ndarray:
    y_phys = np.asarray(y_phys, dtype=float)
    eps = 1e-12
    y_phys = np.maximum(y_phys, eps)
    if base == "10":
        return np.log10(y_phys)
    return np.log(y_phys)

def to_phys(y_log: np.ndarray, base: str = "e") -> np.ndarray:
    y_log = np.asarray(y_log, dtype=float)
    if base == "10":
        return np.power(10.0, y_log)
    return np.exp(y_log)


# ----------------------------
# Metrics (NN-style)
# ----------------------------
def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))

def eval_regression(y_true_log, y_pred_log, log_base="e") -> Dict[str, float]:
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)

    # log-space
    mae_log = float(mean_absolute_error(y_true_log, y_pred_log))
    rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    bias_log = bias(y_true_log, y_pred_log)

    # physical space
    y_true_phys = to_phys(y_true_log, base=log_base)
    y_pred_phys = to_phys(y_pred_log, base=log_base)
    mae_phys = float(mean_absolute_error(y_true_phys, y_pred_phys))
    rmse_phys = float(np.sqrt(mean_squared_error(y_true_phys, y_pred_phys)))
    bias_phys = bias(y_true_phys, y_pred_phys)

    return {
        "mae_log": mae_log,
        "rmse_log": rmse_log,
        "bias_log": bias_log,
        "mae_phys": mae_phys,
        "rmse_phys": rmse_phys,
        "bias_phys": bias_phys,
    }


# ----------------------------
# Splitting (uses NN split if present)
# ----------------------------
def make_train_val_split(
    df: pd.DataFrame,
    cfg: RegConfig,
    test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns train_idx, val_idx.
    Priority:
      1) cfg.split_col if exists with 'train' and 'val' labels
      2) group-based split if cfg.group_col exists
      3) plain seeded random split (optionally stratified by binned y)
    """
    if cfg.split_col and cfg.split_col in df.columns:
        split = df[cfg.split_col].astype(str).str.lower()
        train_idx = np.where(split == "train")[0]
        val_idx = np.where(split == "val")[0]
        if len(train_idx) > 0 and len(val_idx) > 0:
            return train_idx, val_idx
        # if your file has only train/test, still fallback
        # (we keep going)

    if cfg.group_col and cfg.group_col in df.columns:
        # simple deterministic group split:
        # shuffle groups with seed, allocate first (1-test_size) to train.
        rng = np.random.RandomState(cfg.seed)
        groups = df[cfg.group_col].astype(str).values
        uniq = np.unique(groups)
        rng.shuffle(uniq)
        cut = int(round((1.0 - test_size) * len(uniq)))
        train_groups = set(uniq[:cut])
        train_idx = np.where(np.isin(groups, list(train_groups)))[0]
        val_idx = np.where(~np.isin(groups, list(train_groups)))[0]
        return train_idx, val_idx

    # fallback: stratify by binned y (helps stable val distribution)
    y_phys = df[cfg.y_phys_col].values.astype(float)
    y_log = to_log(y_phys, base=cfg.log_base)
    # 10 bins or fewer if small dataset
    n_bins = min(10, max(2, int(len(df) / 20)))
    bins = pd.qcut(y_log, q=n_bins, duplicates="drop")
    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=cfg.seed,
        shuffle=True,
        stratify=bins
    )
    return np.asarray(train_idx), np.asarray(val_idx)


# ----------------------------
# Plots
# ----------------------------
def save_scatter_true_pred(y_true_phys, y_pred_phys, out_path, title):
    plt.figure(figsize=(5, 5), dpi=200)
    plt.scatter(y_true_phys, y_pred_phys, s=10)
    mn = float(min(np.min(y_true_phys), np.min(y_pred_phys)))
    mx = float(max(np.max(y_true_phys), np.max(y_pred_phys)))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True (phys)")
    plt.ylabel("Pred (phys)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def save_residual_hist(residuals, out_path, title):
    plt.figure(figsize=(5, 4), dpi=200)
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual (pred - true) phys")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def save_learning_curve_plot(x_sizes, y_train, y_val, out_path, title, ylab):
    plt.figure(figsize=(6, 4), dpi=200)
    plt.plot(x_sizes, y_train, marker="o", label="Train")
    plt.plot(x_sizes, y_val, marker="s", label="Val")
    plt.xlabel("Training samples")
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ----------------------------
# Models (3 basic regressors)
# ----------------------------
def get_3_reg_models(seed: int = 0) -> Dict[str, object]:
        return {

        # 1) Plain linear regression
        "Linear": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression())
        ]),

        # 2) Ridge
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0))
        ]),

        # 3) PCA + Linear
        "PCA95_Linear": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            ("reg", LinearRegression())
        ]),

        # 4) PCA + Ridge
        "PCA95_Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            ("reg", Ridge(alpha=1.0))
        ]),

        # 5) SVR
        "SVR_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", SVR(kernel="rbf", C=10.0, epsilon=0.05))
        ]),

        # 6) Random Forest
        "RF_300": RandomForestRegressor(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1
        ),
        }


# ----------------------------
# Main evaluation per model
# ----------------------------
def fit_eval_one_model(
    name: str,
    model,
    X_train, y_train_log,
    X_val, y_val_log,
    cfg: RegConfig
) -> Dict[str, object]:
    model.fit(X_train, y_train_log)
    pred_train_log = model.predict(X_train)
    pred_val_log = model.predict(X_val)

    m_train = eval_regression(y_train_log, pred_train_log, log_base=cfg.log_base)
    m_val = eval_regression(y_val_log, pred_val_log, log_base=cfg.log_base)

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": name,
        "seed": cfg.seed,
        "log_base": cfg.log_base,
        "y_phys_col": cfg.y_phys_col,
        "y_log_col": cfg.y_log_col if cfg.y_log_col else "",
        "n_train": int(len(y_train_log)),
        "n_val": int(len(y_val_log)),

        "train_mae_log": m_train["mae_log"],
        "train_rmse_log": m_train["rmse_log"],
        "train_bias_log": m_train["bias_log"],
        "train_mae_phys": m_train["mae_phys"],
        "train_rmse_phys": m_train["rmse_phys"],
        "train_bias_phys": m_train["bias_phys"],

        "val_mae_log": m_val["mae_log"],
        "val_rmse_log": m_val["rmse_log"],
        "val_bias_log": m_val["bias_log"],
        "val_mae_phys": m_val["mae_phys"],
        "val_rmse_phys": m_val["rmse_phys"],
        "val_bias_phys": m_val["bias_phys"],
    }
    return row, pred_val_log


def compute_learning_curves(
    name: str,
    model,
    X_train, y_train_log,
    cfg: RegConfig
) -> Dict[str, str]:
    """
    Learning curves computed on TRAIN ONLY using CV splits inside train.
    We plot:
      - train loss (MSE on log)
      - val MAE log
      - val MAE phys
    """
    # scorers: sklearn expects "higher is better", so we use negative errors.
    def neg_mae_log(y_true, y_pred):
        return -mean_absolute_error(y_true, y_pred)

    def neg_mse_log(y_true, y_pred):
        return -mean_squared_error(y_true, y_pred)

    def neg_mae_phys_from_log(y_true_log, y_pred_log):
        y_true_phys = to_phys(y_true_log, base=cfg.log_base)
        y_pred_phys = to_phys(y_pred_log, base=cfg.log_base)
        return -mean_absolute_error(y_true_phys, y_pred_phys)

    scorer_mae_log = make_scorer(neg_mae_log, greater_is_better=True)
    scorer_mse_log = make_scorer(neg_mse_log, greater_is_better=True)
    scorer_mae_phys = make_scorer(neg_mae_phys_from_log, greater_is_better=True)

    train_sizes = np.array(cfg.lc_train_sizes)

    # 1) MSE(log) => convert to "loss" (positive)
    sizes, tr_mse, va_mse = learning_curve(
        model, X_train, y_train_log,
        train_sizes=train_sizes,
        cv=5,
        scoring=scorer_mse_log,
        n_jobs=-1,
        shuffle=True,
        random_state=cfg.seed
    )
    train_loss = -tr_mse.mean(axis=1)
    val_loss = -va_mse.mean(axis=1)

    # 2) MAE(log)
    _, tr_mae_log, va_mae_log = learning_curve(
        model, X_train, y_train_log,
        train_sizes=train_sizes,
        cv=5,
        scoring=scorer_mae_log,
        n_jobs=-1,
        shuffle=True,
        random_state=cfg.seed
    )
    train_mae_log = -tr_mae_log.mean(axis=1)
    val_mae_log = -va_mae_log.mean(axis=1)

    # 3) MAE(phys)
    _, tr_mae_phys, va_mae_phys = learning_curve(
        model, X_train, y_train_log,
        train_sizes=train_sizes,
        cv=5,
        scoring=scorer_mae_phys,
        n_jobs=-1,
        shuffle=True,
        random_state=cfg.seed
    )
    train_mae_phys = -tr_mae_phys.mean(axis=1)
    val_mae_phys = -va_mae_phys.mean(axis=1)

    out = {}

    os.makedirs(cfg.save_dir, exist_ok=True)

    p1 = os.path.join(cfg.save_dir, f"lc_{name}_trainloss_vs_valloss.png")
    save_learning_curve_plot(
        sizes, train_loss, val_loss,
        p1, title=f"{name} learning curve (log MSE loss)", ylab="MSE loss (log)"
    )
    out["lc_loss_png"] = p1

    p2 = os.path.join(cfg.save_dir, f"lc_{name}_mae_log.png")
    save_learning_curve_plot(
        sizes, train_mae_log, val_mae_log,
        p2, title=f"{name} learning curve (MAE log)", ylab="MAE (log)"
    )
    out["lc_mae_log_png"] = p2

    p3 = os.path.join(cfg.save_dir, f"lc_{name}_mae_phys.png")
    save_learning_curve_plot(
        sizes, train_mae_phys, val_mae_phys,
        p3, title=f"{name} learning curve (MAE phys)", ylab="MAE (phys)"
    )
    out["lc_mae_phys_png"] = p3

    # also save raw curve arrays
    curve_csv = os.path.join(cfg.save_dir, f"lc_{name}_curves.csv")
    pd.DataFrame({
        "train_size": sizes,
        "train_loss_log_mse": train_loss,
        "val_loss_log_mse": val_loss,
        "train_mae_log": train_mae_log,
        "val_mae_log": val_mae_log,
        "train_mae_phys": train_mae_phys,
        "val_mae_phys": val_mae_phys,
    }).to_csv(curve_csv, index=False)
    out["lc_curves_csv"] = curve_csv

    return out
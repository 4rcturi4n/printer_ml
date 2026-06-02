import os
import json
import numpy as np
import pandas as pd


out_dir = "results/dummy_baseline_reg"
split_dir = "data/processed/kfold_splits"


def run_mean_baseline(split_dir, out_dir, n_splits=5):
    os.makedirs(out_dir, exist_ok=True)

    results = []

    for fold in range(1, n_splits + 1):
        train_csv = os.path.join(split_dir, f"fold_{fold}_train.csv")
        val_csv = os.path.join(split_dir, f"fold_{fold}_val.csv")

        df_train = pd.read_csv(train_csv)
        df_val = pd.read_csv(val_csv)

        target_col = "log_axial_resolution"

        if target_col not in df_train.columns:
            raise ValueError(f"{train_csv} does not contain column {target_col}")

        if target_col not in df_val.columns:
            raise ValueError(f"{val_csv} does not contain column {target_col}")

        train_mean_log = df_train[target_col].mean()
        train_std_log = df_train[target_col].std() + 1e-8

        val_true_log = df_val[target_col].to_numpy()

        # Mean baseline in normalized space is 0.
        val_pred_norm = np.zeros_like(val_true_log)

        # Convert normalized prediction back to log space.
        val_pred_log = val_pred_norm * train_std_log + train_mean_log

        error_log = val_pred_log - val_true_log

        val_mae_log = np.mean(np.abs(error_log))
        val_rmse_log = np.sqrt(np.mean(error_log ** 2))
        bias_log = np.mean(error_log)

        val_true_phys = np.exp(val_true_log)
        val_pred_phys = np.exp(val_pred_log)

        error_phys = val_pred_phys - val_true_phys

        val_mae_phys = np.mean(np.abs(error_phys))
        val_rmse_phys = np.sqrt(np.mean(error_phys ** 2))
        bias_phys = np.mean(error_phys)

        results.append(
            {
                "fold": fold,
                "baseline": "train_mean_log",
                "train_rows": len(df_train),
                "val_rows": len(df_val),
                "train_mean_log": train_mean_log,
                "train_std_log": train_std_log,
                "val_mae_log": val_mae_log,
                "val_rmse_log": val_rmse_log,
                "bias_log": bias_log,
                "val_mae_phys": val_mae_phys,
                "val_rmse_phys": val_rmse_phys,
                "bias_phys": bias_phys,
            }
        )

        print(f"✅ Fold {fold}")
        print(f"   train rows: {len(df_train)}")
        print(f"   val rows:   {len(df_val)}")
        print(f"   train mean log: {train_mean_log:.6f}")
        print(f"   val MAE phys:   {val_mae_phys:.6f}")
        print(f"   val RMSE phys:  {val_rmse_phys:.6f}")
        print(f"   bias phys:      {bias_phys:.6f}")
        print()

    results_df = pd.DataFrame(results)

    out_csv = os.path.join(out_dir, "dummy_baseline_reg.csv")
    results_df.to_csv(out_csv, index=False)

    metric_cols = [
        "val_mae_log",
        "val_rmse_log",
        "bias_log",
        "val_mae_phys",
        "val_rmse_phys",
        "bias_phys",
    ]

    summary = {
        "baseline": "train_mean_log",
        "n_splits": n_splits,
        "results_csv": out_csv,
        "metrics": {},
    }

    for col in metric_cols:
        summary["metrics"][col] = {
            "mean": float(results_df[col].mean()),
            "std": float(results_df[col].std()),
        }

    out_json = os.path.join(out_dir, "dummy_baseline_reg_summary.json")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved fold results to: {out_csv}")
    print(f"Saved summary to: {out_json}")

    return results_df, summary


if __name__ == "__main__":
    run_mean_baseline(
        split_dir=split_dir,
        out_dir=out_dir,
        n_splits=5,
    )
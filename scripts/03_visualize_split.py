import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def visualize_split(train_csv, val_csv):
    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)

    # Ensure clean numeric
    for df in [train_df, val_df]:
        df["axial_resolution"] = pd.to_numeric(df["axial_resolution"], errors="coerce")
        df.dropna(subset=["axial_resolution"], inplace=True)
        df["log_axial_resolution"] = np.log(df["axial_resolution"])

    # -------------------------
    # Log space distribution
    # -------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(train_df["log_axial_resolution"], bins=30, alpha=0.6, label="Train")
    plt.hist(val_df["log_axial_resolution"], bins=30, alpha=0.6, label="Validation")
    plt.xlabel("log(Axial resolution)")
    plt.ylabel("Count")
    plt.title("Train vs Validation Distribution (log space)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Raw axial distribution
    # -------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(train_df["axial_resolution"], bins=30, alpha=0.6, label="Train")
    plt.hist(val_df["axial_resolution"], bins=30, alpha=0.6, label="Validation")
    plt.xlabel("Axial resolution")
    plt.ylabel("Count")
    plt.title("Train vs Validation Distribution (raw space)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Print summary stats
    # -------------------------
    print("\nTrain stats:")
    print(train_df["axial_resolution"].describe())

    print("\nValidation stats:")
    print(val_df["axial_resolution"].describe())


if __name__ == "__main__":
    visualize_split(
        train_csv="data/processed/train.csv",
        val_csv="data/processed/val.csv",
    )

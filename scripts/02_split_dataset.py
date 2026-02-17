from printer_ml.split_dataset import split_regression


def main():
    split_regression(
        in_csv="data/processed/video_xl_dataset.csv",
        out_train="data/processed/train.csv",
        out_val="data/processed/val.csv",
        out_split_json="data/processed/split_ids.json",
        seed=42,
        val_size=0.3,
        n_bins=4,
    )


if __name__ == "__main__":
    main()

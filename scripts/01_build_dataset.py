# scripts/01_build_dataset.py
from printer_ml.dataset_maker import build_dataset, FolderPair


def main():
    folder_pairs = [
        FolderPair(
            videos_folder=r"data/Results_10_2025/Results_10_2025/Results/Videos/250617",
            xl_folder=r"data/Results_10_2025/Results_10_2025/Results/ImageJ/1#250617_x10000_21-15@",
            label="250617",
        ),
        FolderPair(
            videos_folder=r"data/Results_10_2025/Results_10_2025/Results/Videos/250624",
            xl_folder=r"data/Results_10_2025/Results_10_2025/Results/ImageJ/2#250624_x10000_15-5@",
            label="250624",
        ),
        FolderPair(
            videos_folder=r"data/Results_10_2025/Results_10_2025/Results/Videos/250627",
            xl_folder=r"data/Results_10_2025/Results_10_2025/Results/ImageJ/2#250627_x10000_15-3@",
            label="250627",
        ),
        FolderPair(
            videos_folder=r"data/Results_10_2025/Results_10_2025/Results/Videos/250710",
            xl_folder=r"data/Results_10_2025/Results_10_2025/Results/ImageJ/2#c250710_x1000_15-5@",
            label="250710",
        ),
        FolderPair(
            videos_folder=r"data/Results_10_2025/Results_10_2025/Results/Videos/250715",
            xl_folder=r"data/Results_10_2025/Results_10_2025/Results/ImageJ/1#250715_x10000_20-10@",
            label="250715",
        ),
    ]

    build_dataset(folder_pairs, out_csv="data/processed/video_xl_dataset.csv")


if __name__ == "__main__":
    main()

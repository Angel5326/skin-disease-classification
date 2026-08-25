"""
HAM10000 Data & Preprocessing Pipeline
Role: Data & Preprocessing Lead - Skin Disease Classification Project

Run in Google Colab or locally with the Kaggle API configured.
Produces: train_split.csv, val_split.csv, test_split.csv, class_weights.json
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

RANDOM_STATE = 42
DATA_DIR = os.path.join("data", "raw")
OUTPUT_DIR = "outputs"

DX_FULL_NAMES = {
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
    "bkl": "Benign keratosis",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses",
    "vasc": "Vascular lesions",
    "df": "Dermatofibroma",
}


def download_dataset():
    """Download and extract HAM10000 from Kaggle. Requires kaggle.json configured."""
    os.system("kaggle datasets download -d kmader/skin-cancer-mnist-ham10000")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.system(f"unzip -q skin-cancer-mnist-ham10000.zip -d {DATA_DIR}")
    print("Dataset downloaded and extracted to:", DATA_DIR)


def load_and_clean_metadata():
    """Load metadata CSV, impute missing values, map image file paths."""
    meta_path = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
    meta = pd.read_csv(meta_path)

    print("Missing values before cleaning:")
    print(meta.isnull().sum())

    meta["age"] = meta["age"].fillna(meta["age"].median())

    img_paths = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(DATA_DIR, "*", "*.jpg"))
    }
    meta["path"] = meta["image_id"].map(img_paths)
    meta = meta.dropna(subset=["path"]).reset_index(drop=True)
    meta["dx_full"] = meta["dx"].map(DX_FULL_NAMES)

    print(f"Final dataset: {meta.shape[0]} images, {meta['lesion_id'].nunique()} unique lesions")
    return meta


def run_eda(meta, save_dir=OUTPUT_DIR):
    """Generate and save EDA plots: class distribution, age/sex, sample images."""
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=meta, y="dx", order=meta["dx"].value_counts().index)
    plt.title("Class Distribution (dx)")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "class_distribution.png"))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(meta["age"], bins=20, ax=axes[0])
    axes[0].set_title("Age Distribution")
    sns.countplot(data=meta, x="sex", ax=axes[1])
    axes[1].set_title("Sex Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "age_sex_distribution.png"))
    plt.close()

    fig, axes = plt.subplots(1, len(meta["dx"].unique()), figsize=(20, 4))
    for ax, cls in zip(axes, meta["dx"].unique()):
        sample_path = meta[meta["dx"] == cls]["path"].iloc[0]
        img = plt.imread(sample_path)
        ax.imshow(img)
        ax.set_title(cls)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sample_images_per_class.png"))
    plt.close()

    print(f"EDA plots saved to {save_dir}/")


def class_balance_analysis(meta, save_dir=OUTPUT_DIR):
    """Compute imbalance ratio and balanced class weights; save to JSON."""
    class_counts = meta["dx"].value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"Imbalance ratio (majority/minority): {imbalance_ratio:.1f}")
    print((class_counts / class_counts.sum() * 100).round(2))

    classes = sorted(meta["dx"].unique())
    labels_int = meta["dx"].astype("category").cat.codes
    weights = compute_class_weight("balanced", classes=np.arange(len(classes)), y=labels_int)
    class_weight_dict = dict(zip(classes, weights.tolist()))

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "class_weights.json"), "w") as f:
        json.dump(class_weight_dict, f, indent=2)

    print("Class weights:", class_weight_dict)
    return class_weight_dict


def create_splits(meta, save_dir=OUTPUT_DIR, test_size=0.30, val_ratio=0.50):
    """
    Split by lesion_id (not image) to prevent leakage of the same lesion
    across train/val/test. Stratified by diagnosis class.
    """
    lesion_df = meta.drop_duplicates("lesion_id")[["lesion_id", "dx"]]

    train_ids, temp_ids = train_test_split(
        lesion_df, test_size=test_size, stratify=lesion_df["dx"], random_state=RANDOM_STATE
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=val_ratio, stratify=temp_ids["dx"], random_state=RANDOM_STATE
    )

    train_df = meta[meta["lesion_id"].isin(train_ids["lesion_id"])].reset_index(drop=True)
    val_df = meta[meta["lesion_id"].isin(val_ids["lesion_id"])].reset_index(drop=True)
    test_df = meta[meta["lesion_id"].isin(test_ids["lesion_id"])].reset_index(drop=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n{name} distribution:\n{df['dx'].value_counts(normalize=True).round(3)}")

    os.makedirs(save_dir, exist_ok=True)
    train_df.to_csv(os.path.join(save_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(save_dir, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "test_split.csv"), index=False)

    return train_df, val_df, test_df


def main():
    if not os.path.isdir(DATA_DIR):
        download_dataset()

    meta = load_and_clean_metadata()
    run_eda(meta)
    class_balance_analysis(meta)
    create_splits(meta)
    print("\nPreprocessing complete. Outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

"""
Augmentation pipeline and Dataset class for the HAM10000 skin lesion images.
Hand this module to the modeling lead once splits are finalized.
"""

import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Applied only to the training set
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.7),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

# Resize + normalize only - no augmentation for validation/test
eval_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for HAM10000. Expects a DataFrame with 'path' and 'dx' columns
    (as produced by data_preprocessing.create_splits).
    """

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.classes = sorted(df["dx"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.class_to_idx[row["dx"]]

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, label


if __name__ == "__main__":
    import pandas as pd

    train_df = pd.read_csv("outputs/train_split.csv")
    val_df = pd.read_csv("outputs/val_split.csv")

    train_dataset = HAM10000Dataset(train_df, transform=train_transform)
    val_dataset = HAM10000Dataset(val_df, transform=eval_transform)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data/raw"
OUTPUT_DIR = "outputs"

MODEL_PATH = "cnn_baseline_best.pth"
FEATURE_PATH = os.path.join(OUTPUT_DIR, "abcde_features.csv")

TRAIN_SPLIT = os.path.join(OUTPUT_DIR, "train_split.csv")
VAL_SPLIT = os.path.join(OUTPUT_DIR, "val_split.csv")
TEST_SPLIT = os.path.join(OUTPUT_DIR, "test_split.csv")

IMG_DIR1 = os.path.join(DATA_DIR, "HAM10000_images_part_1")
IMG_DIR2 = os.path.join(DATA_DIR, "HAM10000_images_part_2")

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

LABEL_MAP = {
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# CNN ARCHITECTURE
# ============================================================

class SkinCancerCNN(nn.Module):

    def __init__(self, num_classes=7):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        features = self.features(x)

        features = features.view(
            features.size(0),
            -1
        )

        return self.classifier(features)


# ============================================================
# HYBRID MODEL
# ============================================================

class HybridModel(nn.Module):

    def __init__(self, cnn):

        super().__init__()

        self.cnn_features = cnn.features

        # Freeze CNN feature extractor initially
        for param in self.cnn_features.parameters():
            param.requires_grad = False

        self.abcd_branch = nn.Sequential(

            nn.Linear(4, 32),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(

            nn.Linear(512 + 32, 256),
            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, 7)
        )

    def forward(self, image, abcd):

        cnn_features = self.cnn_features(
            image
        )

        cnn_features = cnn_features.view(
            cnn_features.size(0),
            -1
        )

        abcd_features = self.abcd_branch(
            abcd
        )

        fused = torch.cat(
            [cnn_features, abcd_features],
            dim=1
        )

        return self.classifier(
            fused
        )


# ============================================================
# IMAGE PATH
# ============================================================

def find_image_path(image_id):

    path1 = os.path.join(
        IMG_DIR1,
        f"{image_id}.jpg"
    )

    if os.path.exists(path1):
        return path1

    path2 = os.path.join(
        IMG_DIR2,
        f"{image_id}.jpg"
    )

    return path2


# ============================================================
# LOAD DATA
# ============================================================

features_df = pd.read_csv(
    FEATURE_PATH
)

feature_columns = [
    "A_asymmetry",
    "B_border_irregularity",
    "C_color_variation",
    "D_diameter_pixels"
]


train_df = pd.read_csv(
    TRAIN_SPLIT
)

val_df = pd.read_csv(
    VAL_SPLIT
)

test_df = pd.read_csv(
    TEST_SPLIT
)


train_df = train_df.merge(
    features_df[
        ["image_id"] + feature_columns
    ],
    on="image_id",
    how="left"
)

val_df = val_df.merge(
    features_df[
        ["image_id"] + feature_columns
    ],
    on="image_id",
    how="left"
)

test_df = test_df.merge(
    features_df[
        ["image_id"] + feature_columns
    ],
    on="image_id",
    how="left"
)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

train_means = train_df[
    feature_columns
].mean()

train_df[
    feature_columns
] = train_df[
    feature_columns
].fillna(
    train_means
)

val_df[
    feature_columns
] = val_df[
    feature_columns
].fillna(
    train_means
)

test_df[
    feature_columns
] = test_df[
    feature_columns
].fillna(
    train_means
)


# ============================================================
# SCALE ABCD FEATURES
# ============================================================

scaler = StandardScaler()

train_df[
    feature_columns
] = scaler.fit_transform(
    train_df[feature_columns]
)

val_df[
    feature_columns
] = scaler.transform(
    val_df[feature_columns]
)

test_df[
    feature_columns
] = scaler.transform(
    test_df[feature_columns]
)


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (128, 128)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# DATASET
# ============================================================

class HybridDataset(Dataset):

    def __init__(
        self,
        dataframe
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

    def __len__(self):

        return len(
            self.df
        )

    def __getitem__(self, idx):

        row = self.df.iloc[
            idx
        ]

        image_path = find_image_path(
            row["image_id"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = transform(
            image
        )

        abcd = torch.tensor(

            row[
                feature_columns
            ].values.astype(
                np.float32
            ),

            dtype=torch.float32
        )

        label = LABEL_MAP[
            row["dx"]
        ]

        return (
            image,
            abcd,
            label
        )


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(

    HybridDataset(
        train_df
    ),

    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(

    HybridDataset(
        val_df
    ),

    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(

    HybridDataset(
        test_df
    ),

    batch_size=64,
    shuffle=False
)


# ============================================================
# LOAD PRETRAINED CNN
# ============================================================

cnn = SkinCancerCNN()

cnn.load_state_dict(

    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

print(
    "CNN weights loaded successfully."
)


# ============================================================
# CREATE HYBRID MODEL
# ============================================================

model = HybridModel(
    cnn
).to(device)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=0.001
)


# ============================================================
# TRAINING
# ============================================================

epochs = 10

best_val_acc = 0.0


print()
print("=" * 60)
print("HYBRID MODEL TRAINING")
print("=" * 60)


for epoch in range(epochs):

    model.train()

    running_loss = 0.0


    for images, abcd, labels in train_loader:

        images = images.to(
            device
        )

        abcd = abcd.to(
            device
        )

        labels = labels.to(
            device
        )


        optimizer.zero_grad()


        outputs = model(
            images,
            abcd
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        running_loss += loss.item()


    train_loss = (
        running_loss /
        len(train_loader)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    all_preds = []
    all_true = []


    with torch.no_grad():

        for images, abcd, labels in val_loader:

            images = images.to(
                device
            )

            abcd = abcd.to(
                device
            )

            labels = labels.to(
                device
            )


            outputs = model(
                images,
                abcd
            )


            preds = torch.argmax(
                outputs,
                dim=1
            )


            all_preds.extend(
                preds.cpu().numpy()
            )

            all_true.extend(
                labels.cpu().numpy()
            )


    val_acc = accuracy_score(
        all_true,
        all_preds
    )


    print(

        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )


    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(

            model.state_dict(),

            os.path.join(
                OUTPUT_DIR,
                "hybrid_model_best.pth"
            )
        )

        print(
            "Best hybrid model saved."
        )


# ============================================================
# TEST
# ============================================================

model.load_state_dict(

    torch.load(

        os.path.join(
            OUTPUT_DIR,
            "hybrid_model_best.pth"
        ),

        map_location=device
    )
)


model.eval()

all_preds = []
all_true = []


with torch.no_grad():

    for images, abcd, labels in test_loader:

        images = images.to(
            device
        )

        abcd = abcd.to(
            device
        )

        labels = labels.to(
            device
        )


        outputs = model(
            images,
            abcd
        )


        preds = torch.argmax(
            outputs,
            dim=1
        )


        all_preds.extend(
            preds.cpu().numpy()
        )

        all_true.extend(
            labels.cpu().numpy()
        )


test_acc = accuracy_score(
    all_true,
    all_preds
)


print()
print("=" * 60)
print("HYBRID MODEL COMPLETE")
print("=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_val_acc * 100:.2f}%"
)

print(
    f"Test Accuracy: "
    f"{test_acc * 100:.2f}%"
)

print(
    "Model saved to: "
    "outputs/hybrid_model_best.pth"
)
# scripts/train_cnn.py
# ============================================
# MEMBER 02: Custom CNN Baseline + Curves
# ============================================
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# 1. CHECK DATA & RECREATE SPLITS IF MISSING
# ----------------------------
def prepare_data():
    if not os.path.exists('data/raw/HAM10000_metadata.csv'):
        raise FileNotFoundError(
            "Dataset not found! Please download HAM10000 from Kaggle and place it in 'data/raw/'\n"
            "Or run: kaggle datasets download -d kmader/skin-cancer-mnist-ham10000"
        )
    
    if not os.path.exists('outputs/train_split.csv'):
        print("⚠️ Splits missing. Regenerating from metadata...")
        meta_df = pd.read_csv('data/raw/HAM10000_metadata.csv')
        unique_lesions = meta_df['lesion_id'].unique()
        train_les, temp_les = train_test_split(unique_lesions, test_size=0.3, random_state=42)
        val_les, test_les = train_test_split(temp_les, test_size=0.5, random_state=42)
        os.makedirs('outputs', exist_ok=True)
        meta_df[meta_df['lesion_id'].isin(train_les)].to_csv('outputs/train_split.csv', index=False)
        meta_df[meta_df['lesion_id'].isin(val_les)].to_csv('outputs/val_split.csv', index=False)
        meta_df[meta_df['lesion_id'].isin(test_les)].to_csv('outputs/test_split.csv', index=False)
        print("✅ Splits recreated.")

prepare_data()

# ----------------------------
# 2. DATA LOADER
# ----------------------------
DATA_DIR = 'data/raw'
IMG_DIR1 = os.path.join(DATA_DIR, 'HAM10000_images_part_1')
IMG_DIR2 = os.path.join(DATA_DIR, 'HAM10000_images_part_2')

def find_image_path(image_id):
    path1 = os.path.join(IMG_DIR1, f"{image_id}.jpg")
    if os.path.exists(path1):
        return path1
    path2 = os.path.join(IMG_DIR2, f"{image_id}.jpg")
    if os.path.exists(path2):
        return path2
    return path1

df_train = pd.read_csv('outputs/train_split.csv')
df_val = pd.read_csv('outputs/val_split.csv')
df_test = pd.read_csv('outputs/test_split.csv')

meta_df = pd.read_csv(os.path.join(DATA_DIR, 'HAM10000_metadata.csv'))
lesion_map = {lesion: idx for idx, lesion in enumerate(sorted(meta_df['dx'].unique()))}
print("🏷️ Label Map:", lesion_map)

for df in [df_train, df_val, df_test]:
    df['image_path'] = df['image_id'].apply(find_image_path)
    df['label'] = df['dx'].map(lesion_map)

print(f"📊 Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

class SkinDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img = Image.open(self.df.iloc[idx]['image_path']).convert('RGB')
        label = self.df.iloc[idx]['label']
        if self.transform:
            img = self.transform(img)
        return img, label

transform = {
    'train': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

train_loader = DataLoader(SkinDataset(df_train, transform['train']), batch_size=64, shuffle=True)
val_loader = DataLoader(SkinDataset(df_val, transform['val']), batch_size=64, shuffle=False)

# ----------------------------
# 3. CNN ARCHITECTURE
# ----------------------------
class SkinCancerCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SkinCancerCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.classifier(x)

# ----------------------------
# 4. TRAINING (Using best hyperparameters from Colab Optuna)
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"💻 Using device: {device}")

model = SkinCancerCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0002835, weight_decay=7.858e-05)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
criterion = nn.CrossEntropyLoss()

train_losses, val_losses, val_accs = [], [], []
epochs = 15  # Set to 25 for final run, 15 is quick for testing

print("\n🔄 Training CNN Baseline...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    model.eval()
    val_loss, all_preds, all_true = 0.0, [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
    val_losses.append(val_loss / len(val_loader))
    acc = accuracy_score(all_true, all_preds)
    val_accs.append(acc)
    scheduler.step(val_loss)
    print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {acc:.4f}')

# Save
torch.save(model.state_dict(), 'cnn_baseline_best.pth')
print("✅ Model saved as cnn_baseline_best.pth")

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('CNN Loss Curves')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(val_accs, label='Val Accuracy', color='green')
plt.title('CNN Accuracy Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('cnn_curves.png')
print("✅ cnn_curves.png saved")
plt.show()

print("\n🎉 MEMBER 02 COMPLETE!")
# scripts/train_tl_ensemble.py
# ============================================
# MEMBER 03: Transfer Learning + Ensemble
# ============================================
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# 1. CHECK DATA & RECREATE SPLITS IF MISSING
# ----------------------------
def prepare_data():
    if not os.path.exists('data/raw/HAM10000_metadata.csv'):
        raise FileNotFoundError(
            "Dataset not found! Please download HAM10000 from Kaggle and place it in 'data/raw/'"
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
# 2. DATA LOADER (224x224 for TL)
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
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

train_loader = DataLoader(SkinDataset(df_train, transform['train']), batch_size=32, shuffle=True)
val_loader = DataLoader(SkinDataset(df_val, transform['val']), batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"💻 Using device: {device}")

# ----------------------------
# 3. TRAINING HELPER
# ----------------------------
def train_tl_model(model, model_name, epochs=10):
    print(f"\n--- 🧠 Fine-tuning {model_name} ---")
    model = model.to(device)
    
    # Freeze all, unfreeze top layers
    for param in model.parameters():
        param.requires_grad = False
    
    if 'resnet' in model_name.lower():
        for param in model.layer4.parameters():
            param.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, 7)
    elif 'efficientnet' in model_name.lower():
        for param in model.features[-3:].parameters():
            param.requires_grad = True
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
    elif 'mobilenet' in model_name.lower():
        for param in model.features[-3:].parameters():
            param.requires_grad = True
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(labels.cpu().numpy())
        acc = accuracy_score(all_true, all_preds)
        scheduler.step(1 - acc)
        print(f'{model_name} Epoch {epoch+1}: Val Acc = {acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'{model_name}_best.pth')
    return model, best_acc

# ----------------------------
# 4. INITIALIZE & TRAIN 3 MODELS
# ----------------------------
models_dict = {
    'EfficientNet-B0': models.efficientnet_b0(weights='DEFAULT'),
    'ResNet50': models.resnet50(weights='DEFAULT'),
    'MobileNetV2': models.mobilenet_v2(weights='DEFAULT')
}

results, trained_models = [], []

for name, model in models_dict.items():
    start = time.time()
    trained_model, acc = train_tl_model(model, name, epochs=10)  # Set to 20+ for final
    trainable = sum(p.numel() for p in trained_model.parameters() if p.requires_grad)
    results.append({
        'Model': name,
        'Val Accuracy': f'{acc:.4f}',
        'Trainable Params': f'{trainable:,}',
        'Time (s)': f'{time.time() - start:.1f}'
    })
    trained_models.append((name, trained_model))

# ----------------------------
# 5. ENSEMBLE
# ----------------------------
print("\n--- 🤝 Building Ensemble ---")
for _, m in trained_models:
    m.eval()

val_true, val_probs = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        val_true.extend(labels.cpu().numpy())
        batch_probs = []
        for _, model in trained_models:
            probs = torch.softmax(model(images), dim=1)
            batch_probs.append(probs.cpu().numpy())
        val_probs.extend(np.mean(batch_probs, axis=0))

ensemble_acc = accuracy_score(val_true, np.argmax(val_probs, axis=1))
print(f"✅ Ensemble Val Accuracy: {ensemble_acc:.4f}")

results.append({
    'Model': 'Ensemble (Avg of 3)',
    'Val Accuracy': f'{ensemble_acc:.4f}',
    'Trainable Params': 'N/A',
    'Time (s)': 'N/A'
})

# ----------------------------
# 6. COMPARISON TABLE
# ----------------------------
df_results = pd.DataFrame(results)
print("\n===== 📊 MODEL COMPARISON TABLE =====")
print(df_results.to_string(index=False))
df_results.to_csv('model_comparison.csv', index=False)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(df_results['Model'][:-1], df_results['Val Accuracy'][:-1].astype(float), color='skyblue')
plt.axhline(y=ensemble_acc, color='red', linestyle='--', label=f'Ensemble = {ensemble_acc:.4f}')
plt.title('Model Comparison: Validation Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

print("\n🎉 MEMBER 03 COMPLETE!")
# src/generate_reports.py
import os, pandas as pd, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = 'data/raw'
IMG_DIR1 = os.path.join(DATA_DIR, 'HAM10000_images_part_1')
IMG_DIR2 = os.path.join(DATA_DIR, 'HAM10000_images_part_2')
device = torch.device('cpu')

def find_path(iid):
    p1 = os.path.join(IMG_DIR1, f"{iid}.jpg")
    p2 = os.path.join(IMG_DIR2, f"{iid}.jpg")
    return p1 if os.path.exists(p1) else p2

df = pd.read_csv('outputs/test_split.csv')
meta = pd.read_csv(os.path.join(DATA_DIR, 'HAM10000_metadata.csv'))
lmap = {l: i for i, l in enumerate(sorted(meta['dx'].unique()))}
df['image_path'] = df['image_id'].apply(find_path)
df['label'] = df['dx'].map(lmap)

class DS(Dataset):
    def __init__(self, d, t): self.d, self.t = d.reset_index(drop=True), t
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        img = Image.open(self.d.iloc[i]['image_path']).convert('RGB')
        return self.t(img), self.d.iloc[i]['label']

t = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
loader = DataLoader(DS(df, t), batch_size=64, shuffle=False)

# CNN Baseline architecture
class SkinCancerCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))

model = SkinCancerCNN()
model.load_state_dict(torch.load('cnn_baseline_best.pth', map_location=device))
model.eval()

preds, trues, probs = [], [], []
with torch.no_grad():
    for x, y in loader:
        o = model(x)
        p = torch.softmax(o, 1)
        preds.extend(o.argmax(1).numpy())
        trues.extend(y.numpy())
        probs.extend(p.numpy())

acc = accuracy_score(trues, preds)
print(f"✅ Test Accuracy: {acc*100:.2f}%")

os.makedirs('reports', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# 1. Classification Report
rep = classification_report(trues, preds, target_names=list(lmap.keys()), output_dict=True)
pd.DataFrame(rep).transpose().to_csv('reports/classification_report.csv')
print("✅ reports/classification_report.csv")

# 2. Confusion Matrix
cm = confusion_matrix(trues, preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=lmap.keys(), yticklabels=lmap.keys())
plt.title(f'Confusion Matrix - CNN Baseline (Acc: {acc*100:.2f}%)')
plt.ylabel('True'); plt.xlabel('Predicted')
plt.tight_layout(); plt.savefig('reports/confusion_matrix_cnn.png', dpi=150); plt.close()
print("✅ reports/confusion_matrix_cnn.png")

# 3. ROC Curves
probs = np.array(probs); trues = np.array(trues)
plt.figure(figsize=(10,8))
for i, c in enumerate(lmap.keys()):
    fpr, tpr, _ = roc_curve(trues == i, probs[:, i])
    plt.plot(fpr, tpr, label=f'{c} (AUC={auc(fpr,tpr):.2f})')
plt.plot([0,1],[0,1],'k--'); plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC Curves - CNN Baseline'); plt.legend(); plt.tight_layout()
plt.savefig('reports/roc_curves_cnn.png', dpi=150); plt.close()
print("✅ reports/roc_curves_cnn.png")

print(f"\n🎉 All reports generated! Test Accuracy: {acc*100:.2f}%")
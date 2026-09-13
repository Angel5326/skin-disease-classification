# src/gradcam.py
# ============================================================
# GRAD-CAM FOR HAM10000 SKIN DISEASE CLASSIFICATION
# Generates 6 sample heatmaps and saves a combined grid image
# ============================================================
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = "data/raw"
MODEL_PATH = "cnn_baseline_best.pth"
OUTPUT_DIR = "outputs"
COMBINED_OUTPUT = os.path.join(OUTPUT_DIR, "gradcam_samples.png")

IMG_DIR1 = os.path.join(DATA_DIR, "HAM10000_images_part_1")
IMG_DIR2 = os.path.join(DATA_DIR, "HAM10000_images_part_2")

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


# ============================================================
# 1. CNN ARCHITECTURE (must match training)
# ============================================================
class SkinCancerCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SkinCancerCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ============================================================
# 2. UTILITIES
# ============================================================
def find_image(image_id):
    path1 = os.path.join(IMG_DIR1, f"{image_id}.jpg")
    if os.path.exists(path1):
        return path1
    path2 = os.path.join(IMG_DIR2, f"{image_id}.jpg")
    if os.path.exists(path2):
        return path2
    raise FileNotFoundError(f"Image not found: {image_id}")


def get_all_image_ids():
    """Return list of all available image_ids from both parts."""
    ids = []
    for d in [IMG_DIR1, IMG_DIR2]:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith(".jpg"):
                    ids.append(f.replace(".jpg", ""))
    return ids


# ============================================================
# 3. LOAD MODEL
# ============================================================
device = torch.device("cpu")
model = SkinCancerCNN(num_classes=7)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("✅ Model loaded successfully.")


# ============================================================
# 4. GRAD-CAM HOOKS
# ============================================================
gradients = None
activations = None

def save_activation(module, input, output):
    global activations
    activations = output

def save_gradient(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

# Hook into last conv layer (features[16] = Conv2d(256, 512))
target_layer = model.features[16]
target_layer.register_forward_hook(save_activation)
target_layer.register_full_backward_hook(save_gradient)


# ============================================================
# 5. PREPROCESSING
# ============================================================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============================================================
# 6. GRAD-CAM GENERATOR (returns original, cam, overlay)
# ============================================================
def generate_gradcam(image_id):
    global gradients, activations

    image_path = find_image(image_id)
    original_image = Image.open(image_path).convert("RGB")
    original_array = np.array(original_image)

    input_tensor = transform(original_image).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    # Forward
    model.zero_grad()
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(output, dim=1).item()
    predicted_probability = probabilities[0, predicted_class].item()

    # Backward
    score = output[0, predicted_class]
    score.backward()

    # CAM computation
    activation = activations.detach()
    gradient = gradients.detach()
    weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activation, dim=1)
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().numpy()

    # Normalize
    cam = cam - cam.min()
    if cam.max() != 0:
        cam = cam / cam.max()

    # Resize to original image size
    height, width = original_array.shape[:2]
    cam = cv2.resize(cam, (width, height))

    # Heatmap
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = 0.4 * heatmap + 0.6 * original_array
    overlay = np.uint8(overlay)

    return original_array, cam, overlay, CLASS_NAMES[predicted_class], predicted_probability * 100


# ============================================================
# 7. RUN FOR 6 SAMPLE IMAGES & SAVE COMBINED GRID
# ============================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_ids = get_all_image_ids()
    if not all_ids:
        raise FileNotFoundError("No HAM10000 images found.")

    # Pick 6 evenly spaced samples for diversity
    num_samples = 6
    step = max(1, len(all_ids) // num_samples)
    sample_ids = all_ids[::step][:num_samples]

    print(f"\n🔍 Generating Grad-CAM for {num_samples} images...")

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i, image_id in enumerate(sample_ids):
        try:
            original, cam, overlay, label, conf = generate_gradcam(image_id)
            print(f"  [{i+1}/{num_samples}] {image_id} → {label} ({conf:.1f}%)")

            # Original
            axes[i, 0].imshow(original)
            axes[i, 0].set_title(f"Original: {image_id}", fontsize=10)
            axes[i, 0].axis("off")

            # Grad-CAM heatmap
            axes[i, 1].imshow(cam, cmap="jet")
            axes[i, 1].set_title("Grad-CAM Heatmap", fontsize=10)
            axes[i, 1].axis("off")

            # Overlay with prediction
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title(f"Pred: {label} ({conf:.1f}%)", fontsize=10)
            axes[i, 2].axis("off")

        except Exception as e:
            print(f"  ⚠️ Skipped {image_id}: {e}")
            for j in range(3):
                axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(COMBINED_OUTPUT, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n✅ Combined Grad-CAM grid saved: {COMBINED_OUTPUT}")
    print(f"🎉 Grad-CAM generation complete!")
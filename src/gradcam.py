import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms


# ============================================================
# GRAD-CAM FOR HAM10000 SKIN DISEASE CLASSIFICATION
# ============================================================

DATA_DIR = "data/raw"
MODEL_PATH = "cnn_baseline_best.pth"
OUTPUT_DIR = "outputs/gradcam"

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


# ============================================================
# 1. SAME CNN ARCHITECTURE USED DURING TRAINING
# ============================================================

class SkinCancerCNN(nn.Module):

    def __init__(self, num_classes=7):

        super(SkinCancerCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

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

        x = self.features(x)

        x = x.view(x.size(0), -1)

        return self.classifier(x)


# ============================================================
# 2. FIND IMAGE
# ============================================================

def find_image(image_id):

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

    if os.path.exists(path2):
        return path2

    raise FileNotFoundError(
        f"Image not found: {image_id}"
    )


# ============================================================
# 3. LOAD MODEL
# ============================================================

device = torch.device("cpu")

model = SkinCancerCNN(num_classes=7)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()

print("Model loaded successfully.")


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


# Last convolutional layer
target_layer = model.features[16]

target_layer.register_forward_hook(
    save_activation
)

target_layer.register_full_backward_hook(
    save_gradient
)


# ============================================================
# 5. IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 6. GRAD-CAM FUNCTION
# ============================================================

def generate_gradcam(image_id):

    global gradients
    global activations

    image_path = find_image(image_id)

    print(f"Using image: {image_path}")


    # Load original image
    original_image = Image.open(
        image_path
    ).convert("RGB")


    # Prepare image
    input_tensor = transform(
        original_image
    ).unsqueeze(0)

    input_tensor = input_tensor.to(device)

    input_tensor.requires_grad_(True)


    # Forward pass
    model.zero_grad()

    output = model(input_tensor)

    probabilities = torch.softmax(
        output,
        dim=1
    )


    # Predicted class
    predicted_class = torch.argmax(
        output,
        dim=1
    ).item()

    predicted_probability = probabilities[
        0,
        predicted_class
    ].item()


    print(
        f"Prediction: {CLASS_NAMES[predicted_class]}"
    )

    print(
        f"Confidence: {predicted_probability * 100:.2f}%"
    )


    # Backward pass
    score = output[
        0,
        predicted_class
    ]

    score.backward()


    # Get activations and gradients
    activation = activations.detach()

    gradient = gradients.detach()


    # Global average pooling of gradients
    weights = torch.mean(
        gradient,
        dim=(2, 3),
        keepdim=True
    )


    # Weighted activation maps
    cam = torch.sum(
        weights * activation,
        dim=1
    )


    cam = torch.relu(cam)


    # Convert to numpy
    cam = cam.squeeze().cpu().numpy()


    # Normalize CAM
    cam = cam - cam.min()

    if cam.max() != 0:

        cam = cam / cam.max()


    # Resize CAM
    original_array = np.array(
        original_image
    )

    height, width = original_array.shape[:2]

    cam = cv2.resize(
        cam,
        (width, height)
    )


    # Create heatmap
    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )


    # Overlay
    overlay = (
        0.4 * heatmap +
        0.6 * original_array
    )

    overlay = np.uint8(
        overlay
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{image_id}_gradcam.png"
    )


    plt.figure(
        figsize=(12, 4)
    )


    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        original_array
    )

    plt.title("Original Image")

    plt.axis("off")


    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        cam,
        cmap="jet"
    )

    plt.title("Grad-CAM")

    plt.axis("off")


    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        overlay
    )

    plt.title(
        f"{CLASS_NAMES[predicted_class]} "
        f"({predicted_probability * 100:.1f}%)"
    )

    plt.axis("off")


    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"Grad-CAM saved: {output_path}"
    )


# ============================================================
# 7. RUN WITH ONE HAM10000 IMAGE
# ============================================================

if __name__ == "__main__":

    # Automatically select first available image
    image_files = []

    if os.path.exists(IMG_DIR1):

        image_files = [
            f for f in os.listdir(IMG_DIR1)
            if f.lower().endswith(".jpg")
        ]

    if not image_files and os.path.exists(IMG_DIR2):

        image_files = [
            f for f in os.listdir(IMG_DIR2)
            if f.lower().endswith(".jpg")
        ]


    if not image_files:

        raise FileNotFoundError(
            "No HAM10000 images found."
        )


    image_id = image_files[0].replace(
        ".jpg",
        ""
    )


    print(
        f"\nRunning Grad-CAM for: {image_id}"
    )

    generate_gradcam(image_id)
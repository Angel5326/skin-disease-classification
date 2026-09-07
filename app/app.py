import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Skin Disease Classification",
    page_icon="🩺",
    layout="centered"
)

# ----------------------------
# Constants
# ----------------------------
MODEL_PATH = "cnn_baseline_best.pth"
CLASS_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_FULL_NAMES = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Model Architecture
# ----------------------------
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


# ----------------------------
# Image preprocessing
# ----------------------------
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ----------------------------
# Model loading (cached)
# ----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file '{MODEL_PATH}' not found. Please place it in the app directory."

    try:
        model = SkinCancerCNN(num_classes=len(CLASS_LABELS))
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

        # Handle both raw state_dict and checkpoint-wrapped state_dict
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


# ----------------------------
# Prediction function
# ----------------------------
def predict(image: Image.Image, model):
    image = image.convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    predicted_label = CLASS_LABELS[predicted_idx.item()]
    confidence_pct = confidence.item() * 100

    all_probs = {
        CLASS_LABELS[i]: probabilities[i].item() * 100
        for i in range(len(CLASS_LABELS))
    }

    return predicted_label, confidence_pct, all_probs


# ----------------------------
# UI - Header
# ----------------------------
st.title("Skin Disease Classification from Dermoscopic Images")
st.markdown(
    "Upload a dermoscopic image of a skin lesion to get an AI-powered "
    "classification prediction across 7 common skin disease categories."
)
st.divider()

# ----------------------------
# Load model
# ----------------------------
model, error_message = load_model()

if error_message:
    st.error(f"⚠️ {error_message}")
    st.info(
        "Make sure the trained model file `cnn_baseline_best.pth` is present "
        "in the same directory as this app before running predictions."
    )

# ----------------------------
# UI - Image uploader
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload a dermoscopic image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    except Exception as e:
        st.error(f"Could not open the uploaded image: {str(e)}")
        image = None

    st.divider()

    predict_clicked = st.button(
        "🔍 Predict",
        use_container_width=True,
        disabled=(model is None)
    )

    if predict_clicked:
        if model is None:
            st.error("Cannot run prediction because the model is not loaded.")
        elif image is None:
            st.error("Cannot run prediction because the image could not be read.")
        else:
            with st.spinner("Analyzing image..."):
                try:
                    predicted_label, confidence_pct, all_probs = predict(image, model)
                    full_name = CLASS_FULL_NAMES.get(predicted_label, predicted_label)

                    st.success("Prediction complete")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Class", predicted_label.upper())
                    with col2:
                        st.metric("Confidence", f"{confidence_pct:.2f}%")

                    st.markdown(f"**Full Diagnosis Name:** {full_name}")

                    st.markdown("#### Class Probability Breakdown")
                    sorted_probs = dict(
                        sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                    )
                    for label, prob in sorted_probs.items():
                        name = CLASS_FULL_NAMES.get(label, label)
                        st.write(f"**{label.upper()}** — {name}")
                        st.progress(min(int(prob), 100), text=f"{prob:.2f}%")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")

else:
    st.info("Please upload an image to begin.")

# ----------------------------
# Disclaimer
# ----------------------------
st.divider()
st.warning("⚠️ This prediction is AI-generated and not a medical diagnosis.")

st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.85em;'>"
    "Skin Disease Classification System | Powered by PyTorch & Streamlit"
    "</div>",
    unsafe_allow_html=True
)

@"
# Skin Disease Classification from Dermoscopic Images

An AI-powered system that classifies skin conditions (eczema, melanoma, acne,
psoriasis) from dermoscopic images using a hybrid CNN + handcrafted-feature
approach, with explainable predictions via Grad-CAM.

## Team
| Name | Role |
|---|---|
| Akash A | Data & Preprocessing Lead |
| Member 2 | Core CNN Model Engineer |
| Member 3 | Transfer Learning & Ensemble Engineer |
| Member 4 | Explainability & Feature Engineering |
| Member 5 | Application/Deployment Engineer |
| Member 6 | Evaluation, Docs & Presentation Lead |

## Dataset
HAM10000 (Kaggle): https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
Download and place under data/raw/ locally (not committed to git).

## Setup
``````bash
git clone https://github.com/Angel5326/skin-disease-classification.git
cd skin-disease-classification
pip install -r requirements.txt
``````

## Project Structure
- data/raw, data/processed — datasets (gitignored)
- notebooks/ — exploration notebooks
- src/ — data_loader.py, train.py, gradcam.py, models/
- app/ — Streamlit deployment app
- reports/ — figures, confusion matrices, final report
- docs/ — per-member notes, meeting logs

## Running the app
``````bash
streamlit run app/app.py
``````
"@ | Out-File -FilePath README.md -Encoding utf8
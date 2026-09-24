# Explainable AI Image Classification using CNN & DeepSHAP

An end-to-end Explainable AI project using a TensorFlow/Keras CNN trained on CIFAR-10, with SHAP DeepExplainer for visual model attribution and a Streamlit interface for interactive inference.

## Workflow

```text
CIFAR-10
   ↓
Train / Validation / Test Split
   ↓
Normalization
   ↓
CNN Training
   ↓
Evaluation
   ↓
Prediction
   ↓
DeepSHAP Explanation
   ↓
Streamlit Web App
```

## Classes

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Repository

```text
Explainable-AI-CIFAR10/
├── app/app.py
├── src/train.py
├── src/predict.py
├── src/explain.py
├── notebooks/Explainable_AI_CIFAR10_DeepSHAP.ipynb
├── images/
├── outputs/
├── models/
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

Install:
```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

This creates the trained Keras model and evaluation outputs.

## Run the application

```bash
streamlit run app/app.py
```

Then open the local URL shown by Streamlit, normally `http://localhost:8501`.

## Explainability

DeepSHAP is used to calculate feature attributions for model predictions. The visualization is a model-attribution explanation; it should not be interpreted as causal or human-like reasoning.

## Important limitation

This is a CIFAR-10 benchmark computer-vision model. It is not a general-purpose image classifier. An uploaded image outside the CIFAR-10 distribution may still receive one of the ten class labels.

## Future improvements

- Data augmentation
- Higher-resolution dataset
- Grad-CAM comparison
- FastAPI inference API
- Docker deployment
- Model monitoring
- Cloud deployment

## Screenshots

Add your training graphs, confusion matrix and DeepSHAP screenshots to `images/`.

## Author

Add your name, GitHub and LinkedIn links here.

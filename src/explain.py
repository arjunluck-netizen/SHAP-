from pathlib import Path
import numpy as np
import shap
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT/"models"/"cifar10_cnn_final.keras"

def create_background(size=50):
    (x_train,_),_ = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32")/255.0
    rng = np.random.default_rng(42)
    return x_train[rng.choice(len(x_train),size=size,replace=False)]

def explain_image(image_array,background=None,model=None):
    model = model or tf.keras.models.load_model(MODEL_PATH)
    background = background if background is not None else create_background()
    explainer = shap.DeepExplainer(model,background)
    return explainer.shap_values(image_array)

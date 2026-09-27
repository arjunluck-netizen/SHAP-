from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "cifar10_cnn_final.keras"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((32, 32))

    image_array = np.asarray(image).astype("float32") / 255.0

    return np.expand_dims(image_array, axis=0)


def predict_image(image, model):
    processed_image = preprocess_image(image)

    probabilities = model.predict(
        processed_image,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))

    return {
        "class": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": probabilities
    }

from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

CLASS_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
ROOT = Path(__file__).resolve().parents[1]
# MODEL_PATH = ROOT/"models"/"cifar10_cnn_final.keras"


# Modify your load_model function:
def load_model():
    model_path = "cifar10_cnn_final.keras"  # or path to your .h5 file
    
    # Force compile=False to avoid custom layer/loss deserialization errors
    return tf.keras.models.load_model(model_path, compile=False)

def load_model(model_path=MODEL_PATH):
    return tf.keras.models.load_model(model_path)

def preprocess_image(image):
    image = image.convert("RGB").resize((32,32))
    arr = np.asarray(image,dtype=np.float32)/255.0
    return np.expand_dims(arr,axis=0)

def predict_image(image, model=None):
    model = model or load_model()
    batch = preprocess_image(image)
    probabilities = model.predict(batch,verbose=0)[0]
    index = int(np.argmax(probabilities))
    return {"class_name":CLASS_NAMES[index],"class_index":index,
            "confidence":float(probabilities[index]),
            "probabilities":probabilities,"input_array":batch}

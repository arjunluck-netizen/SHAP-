from pathlib import Path
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
BATCH_SIZE = 64
EPOCHS = 30
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def main():
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    (x_full, y_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_full, y_test = y_full.ravel(), y_test.ravel()

    x_train, x_val, y_train, y_val = train_test_split(
        x_full, y_full, test_size=5000, random_state=SEED, stratify=y_full
    )
    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32,32,3)),
        tf.keras.layers.Conv2D(32,(3,3),activation="relu",padding="same"),
        tf.keras.layers.Conv2D(32,(3,3),activation="relu",padding="same"),
        tf.keras.layers.MaxPooling2D((2,2)), tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"),
        tf.keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"),
        tf.keras.layers.MaxPooling2D((2,2)), tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10,activation="softmax")
    ])
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=2,min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(MODEL_DIR/"cifar10_cnn_best.keras",monitor="val_accuracy",save_best_only=True)
    ]
    history = model.fit(x_train,y_train,validation_data=(x_val,y_val),
                        epochs=EPOCHS,batch_size=BATCH_SIZE,callbacks=callbacks)

    model.save(MODEL_DIR/"cifar10_cnn_final.keras")
    loss, accuracy = model.evaluate(x_test,y_test,batch_size=BATCH_SIZE)
    print(f"Test accuracy: {accuracy:.4f}")

    probabilities = model.predict(x_test,batch_size=BATCH_SIZE)
    predictions = probabilities.argmax(axis=1)
    report = classification_report(y_test,predictions,target_names=CLASS_NAMES,digits=4)
    print(report)
    (OUTPUT_DIR/"classification_report.txt").write_text(report,encoding="utf-8")
    pd.DataFrame(history.history).to_csv(OUTPUT_DIR/"training_history.csv",index=False)

    cm = confusion_matrix(y_test,predictions)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm,annot=True,fmt="d",xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES)
    plt.title("CIFAR-10 Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR/"confusion_matrix.png",dpi=150); plt.close()

if __name__ == "__main__":
    main()

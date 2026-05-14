import json
import os
import numpy as np
import tensorflow as tf
from classifier import CIFARAdvancedClassifier

MODEL_PATH = "models/cnn_advanced_model.keras"
DATA_PATH = "data/cifar10.npz"

def main():
    data = np.load(DATA_PATH)
    x_train, y_train = data["x_train"], data["y_train"]

    clf = CIFARAdvancedClassifier()
    clf.model = clf.build_model()
    clf.model.summary()

    # Callbacks definition:
    # Stop training if val_loss doesn't improve for 7 epochs, and restore best weights
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=7, 
        restore_best_weights=True
    )
    
    # Save the best model based on validation accuracy
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    print("\n--- Starting Task 5 ---")
    history = clf.model.fit(
        x_train, y_train,
        epochs=50, 
        batch_size=128,
        validation_split=0.1,
        callbacks=[early_stop, checkpoint]
    )
    os.makedirs("results", exist_ok=True)
    with open("results/cnn_advanced_history.json", "w") as f:
        json.dump(history.history, f)
    print(f"\n Successfully saved the best model at: {MODEL_PATH}")

if __name__ == "__main__":
    main()
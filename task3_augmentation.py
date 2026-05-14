"""
Task 3 - Data Augmentation
"""

from __future__ import annotations
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from classifier import CIFARCNNClassifier

# ── Constants ────────────────────────────────────────────────────────────────
DATA_PATH    = "data/cifar10.npz"
MODEL_DIR    = "models"
RESULTS_DIR  = "results"
FIGURES_DIR  = "figures"

EPOCHS       = 30
BATCH_SIZE   = 128

# ── Helper Functions ─────────────────────────────────────────────────────────

def load_history(model_type: str) -> dict:
    path = os.path.join(RESULTS_DIR, f"{model_type}_history.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find history at {path}. Please run 'python main.py train --model cnn' first!")
    with open(path, "r") as f:
        return json.load(f)

def build_augmentation_layer() -> tf.keras.Sequential:
    """Create augmentation pipeline."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ], name="augmentation")

def train_with_augmentation(x_train, y_train):
    print("\n" + "="*60 + "\nTraining CNN WITH augmentation\n" + "="*60)
    
    classifier = CIFARCNNClassifier()
    base_model = classifier.build_model()
    
    augmented_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        build_augmentation_layer(),
        *base_model.layers
    ])
    
    augmented_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    classifier.model = augmented_model

    history = classifier.train(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    classifier.save(os.path.join(MODEL_DIR, "cnn_aug_model.keras"))
    with open(os.path.join(RESULTS_DIR, "cnn_aug_history.json"), "w") as f:
        json.dump(history.history, f)
        
    return history.history

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(h_no_aug: dict, h_aug: dict):
    """ Plot Learning Curves."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    epochs_range = range(1, EPOCHS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Comparison of Baseline CNN vs Augmented CNN", fontsize=16)

    # Plot Accuracy
    axes[0].plot(epochs_range, h_no_aug['accuracy'], 'b--', label='Baseline Train Acc')
    axes[0].plot(epochs_range, h_no_aug['val_accuracy'], 'b', label='Baseline Val Acc')
    axes[0].plot(epochs_range, h_aug['accuracy'], 'r--', label='Augmented Train Acc')
    axes[0].plot(epochs_range, h_aug['val_accuracy'], 'r', label='Augmented Val Acc')
    axes[0].set_title("Accuracy Comparison")
    axes[0].legend()
    axes[0].grid(True)

    # Plot Loss
    axes[1].plot(epochs_range, h_no_aug['loss'], 'b--', label='Baseline Train Loss')
    axes[1].plot(epochs_range, h_no_aug['val_loss'], 'b', label='Baseline Val Loss')
    axes[1].plot(epochs_range, h_aug['loss'], 'r--', label='Augmented Train Loss')
    axes[1].plot(epochs_range, h_aug['val_loss'], 'r', label='Augmented Val Loss')
    axes[1].set_title("Loss Comparison")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "task3_comparison.png"))
    plt.show()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    data = np.load(DATA_PATH)
    x_train, y_train = data["x_train"], data["y_train"]
    
    print("Loading history of Baseline CNN...")
    history_no_aug = load_history("cnn")
    history_aug = train_with_augmentation(x_train, y_train)
    plot_comparison(history_no_aug, history_aug)

if __name__ == "__main__":
    main()
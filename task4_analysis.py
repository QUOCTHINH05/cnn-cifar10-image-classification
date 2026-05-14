import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from classifier import CIFARMLPClassifier, CIFARCNNClassifier

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
MODEL_DIR   = "models"
DATA_PATH   = "data/cifar10.npz"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"Cannot find {path}. Please run train first!")
        return None
    with open(path, "r") as f:
        return json.load(f)

def plot_learning_curves(history, model_name):
    """Plot learning curves from history."""
    if not history: return
    
    epochs = range(1, len(history['loss']) + 1)
    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], 'bo-', label='Train Acc')
    plt.plot(epochs, history['val_accuracy'], 'ro-', label='Val Acc')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['loss'], 'bo-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Val Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"task4_{model_name.lower()}_curves.png"))
    plt.show()

def perform_error_analysis(model_path, x_test, y_test, model_label):
    """Create Confusion Matrix and find top 3 errors."""
    print(f"\n--- Error Analysis for {model_label} ---")
    
    # Load model depending on type
    if "mlp" in model_label.lower():
        clf = CIFARMLPClassifier()
    else:
        clf = CIFARCNNClassifier()
    
    clf.load(model_path)
    y_pred = np.argmax(clf.model.predict(x_test), axis=1)

    # Classification Report
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0) 
    
    confused_indices = np.unravel_index(np.argsort(cm_copy.ravel())[-3:], cm_copy.shape)
    
    print(f"Top 3 most confused classes for {model_label}:")
    for i in range(2, -1, -1):
        true_idx = confused_indices[0][i]
        pred_idx = confused_indices[1][i]
        count = cm[true_idx, pred_idx]
        print(f"{3-i}. Actual is '{CLASS_NAMES[true_idx]}' but predicted as '{CLASS_NAMES[pred_idx]}' ({count} times)")

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    data = np.load(DATA_PATH)
    x_test, y_test = data["x_test"], data["y_test"]

    mlp_hist = load_json("cifar_mlp_history.json")
    plot_learning_curves(mlp_hist, "MLP")

    cnn_hist = load_json("cnn_history.json")
    plot_learning_curves(cnn_hist, "CNN_Baseline")

    perform_error_analysis(os.path.join(MODEL_DIR, "cifar_mlp_model.keras"), x_test, y_test, "MLP")
    perform_error_analysis(os.path.join(MODEL_DIR, "cnn_model.keras"), x_test, y_test, "CNN_Baseline")

if __name__ == "__main__":
    main()
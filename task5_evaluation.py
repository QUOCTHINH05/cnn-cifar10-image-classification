import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from classifier import CIFARAdvancedClassifier

HISTORY_PATH = "results/cnn_advanced_history.json"
MODEL_PATH   = "models/cnn_advanced_model.keras"
DATA_PATH    = "data/cifar10.npz"
FIGURES_DIR  = "figures"
CLASS_NAMES  = ["airplane", "automobile", "bird", "cat", "deer", 
                "dog", "frog", "horse", "ship", "truck"]

def plot_task5_learning_curves():
    if not os.path.exists(HISTORY_PATH):
        print(f"Cannot find history file at {HISTORY_PATH}")
        return

    with open(HISTORY_PATH, "r") as f:
        h = json.load(f)
    
    epochs = range(1, len(h['loss']) + 1)
    plt.figure(figsize=(14, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, h['accuracy'], 'g-', label='Train Accuracy')
    plt.plot(epochs, h['val_accuracy'], 'r-', label='Val Accuracy')
    plt.title('Task 5: Advanced CNN Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, h['loss'], 'g-', label='Train Loss')
    plt.plot(epochs, h['val_loss'], 'r-', label='Val Loss')
    plt.title('Task 5: Advanced CNN Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "task5_learning_curves.png"), dpi=150)
    plt.show()

def plot_task5_confusion_matrix():
    data = np.load(DATA_PATH)
    x_test, y_test = data["x_test"], data["y_test"]

    clf = CIFARAdvancedClassifier()
    clf.load(MODEL_PATH)
    
    print("Evaluating model on test set...")
    y_pred = np.argmax(clf.model.predict(x_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix: Task 5 Advanced CNN")
    plt.ylabel('Actual (True Label)')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "task5_confusion_matrix.png"), dpi=150)
    plt.show()

if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_task5_learning_curves()
    plot_task5_confusion_matrix()
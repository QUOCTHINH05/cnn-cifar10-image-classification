from __future__ import annotations

import abc
from typing import Optional

import numpy as np
import tensorflow as tf


class MNISTClassifier(abc.ABC):
    """Base class for MNIST digit classifiers."""

    def __init__(self):
        self.model: Optional[tf.keras.Model] = None

    @abc.abstractmethod
    def build_model(self) -> tf.keras.Model:
        """Build and return a compiled Keras model."""

    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 10, batch_size: int = 128,
              validation_split: float = 0.1) -> tf.keras.callbacks.History:
        """Train the model on the given data.

        TODO: Implement this method.
        - If self.model is None, call self.build_model() to create it.
        - Use the model's fit() method with the provided parameters.
        - Return the History object from fit().
        """
        if self.model is None:
          self.model = self.build_model()
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return history
      
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model on the test data.

        TODO: Implement this method.
        - Raise RuntimeError if self.model is None.
        - Use the model's evaluate() method to get loss and accuracy.
        - Use the model's predict() method and np.argmax to get predicted labels.
        - Return a dict with keys: "loss", "accuracy", "y_pred".
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized. Call build_model() or train() first.")
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        y_pred = np.argmax(self.model.predict(x_test), axis=1)
        return {"loss": loss, "accuracy": accuracy, "y_pred": y_pred}

    def save(self, path: str) -> None:
        """Save the model to the given file path.

        TODO: Implement this method.
        - Raise RuntimeError if self.model is None.
        - Use the model's save() method.
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized. Call build_model() or train() first.")
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load a model from the given file path.

        TODO: Implement this method.
        - Use tf.keras.models.load_model() and assign to self.model.
        """
        self.model = tf.keras.models.load_model(path)


class LogisticRegressionClassifier(MNISTClassifier):
    """Logistic regression (single dense layer with softmax)."""

    def build_model(self) -> tf.keras.Model:
        """Build a logistic regression model for MNIST.

        TODO: Implement this method.
        - Create a Sequential model with:
          - Input layer accepting 784-dimensional vectors.
          - A single Dense output layer with 10 units and softmax activation.
        - Compile with optimizer="sgd", loss="sparse_categorical_crossentropy",
          and metrics=["accuracy"].
        - Return the compiled model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(784,)),
            tf.keras.layers.Dense(10, activation="softmax")
        ])
        model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model


class NeuralNetworkClassifier(MNISTClassifier):
    """Simple feedforward neural network."""

    def build_model(self) -> tf.keras.Model:
        """Build a simple neural network for MNIST.

        TODO: Implement this method.
        - Create a Sequential model with:
          - Input layer accepting 784-dimensional vectors.
          - Dense hidden layer with 128 units and ReLU activation.
          - Dense hidden layer with 64 units and ReLU activation.
          - Dense output layer with 10 units and softmax activation.
        - Compile with optimizer="adam", loss="sparse_categorical_crossentropy",
          and metrics=["accuracy"].
        - Return the compiled model.
        """
        
        model = tf.keras.Sequential([
          tf.keras.layers.Input(shape=(784,)),
          tf.keras.layers.Dense(128, activation="relu"),
          tf.keras.layers.Dense(64, activation="relu"),
          tf.keras.layers.Dense(10, activation="softmax")
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model


class CIFARMLPClassifier(MNISTClassifier):
    """Baseline MLP for CIFAR-10 classification."""

    def build_model(self) -> tf.keras.Model:
        """Build a baseline MLP model for CIFAR-10.

        TODO: Implement this method.
        - Create a Sequential model with:
          - Input layer accepting 32x32x3 images.
          - Flatten layer.
          - Dense hidden layer with 512 units and ReLU activation.
          - Dense hidden layer with 256 units and ReLU activation.
          - Dense output layer with 10 units and softmax activation.
        - Compile with optimizer="adam", loss="sparse_categorical_crossentropy",
          and metrics=["accuracy"].
        - Return the compiled model.
        """
        model = tf.keras.Sequential([
          tf.keras.layers.Input(shape=(32, 32, 3)),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(512, activation="relu"),
          tf.keras.layers.Dense(256, activation="relu"),
          tf.keras.layers.Dense(10, activation="softmax")
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model


class CIFARCNNClassifier(MNISTClassifier):
    """Convolutional neural network for CIFAR-10 classification."""

    def build_model(self) -> tf.keras.Model:
        """Build a CNN model for CIFAR-10.

        TODO: Implement this method.
        - Create a Sequential model with the following blocks:
          - Input layer accepting 32x32x3 images.
          Block 1:
          - Conv2D(32, 3, padding="same", activation="relu") + BatchNormalization
          - Conv2D(32, 3, padding="same", activation="relu") + BatchNormalization
          - MaxPooling2D(2) + Dropout(0.2)
          Block 2:
          - Conv2D(64, 3, padding="same", activation="relu") + BatchNormalization
          - Conv2D(64, 3, padding="same", activation="relu") + BatchNormalization
          - MaxPooling2D(2) + Dropout(0.3)
          Block 3:
          - Conv2D(128, 3, padding="same", activation="relu") + BatchNormalization
          - Conv2D(128, 3, padding="same", activation="relu") + BatchNormalization
          - MaxPooling2D(2) + Dropout(0.4)
          Head:
          - Flatten
          - Dense(256, activation="relu") + BatchNormalization + Dropout(0.5)
          - Dense(10, activation="softmax")
        - Compile with optimizer="adam", loss="sparse_categorical_crossentropy",
          and metrics=["accuracy"].
        - Return the compiled model.
        """
        model = tf.keras.Sequential([
          tf.keras.layers.Input(shape=(32, 32, 3)),
          
          #Block 1
          tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.MaxPooling2D(2),
          tf.keras.layers.Dropout(0.25),
          
          #Block 2
          tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.MaxPooling2D(2),
          tf.keras.layers.Dropout(0.25),
          
      
          #Head
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(256, activation="relu"),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dropout(0.5),
          tf.keras.layers.Dense(10, activation="softmax")
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model


class CIFARAdvancedClassifier(MNISTClassifier):
    """Deeper CNN with advanced techniques for CIFAR-10 classification."""

    def build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3)),
            
            # Block 1
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.2),
            
            # Block 2
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.3),
            
            # Block 3
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.4),

            # Block 4 
            tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(), 
            
            # Head
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax")
        ])
        
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model
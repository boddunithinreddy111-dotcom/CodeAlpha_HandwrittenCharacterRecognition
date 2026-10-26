"""
Deep Learning Model Architectures for Handwritten Character Recognition.
Includes Deep Neural Network / CNN classifier with softmax probabilities
and serialization support.
"""

from pathlib import Path
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class DeepCharacterClassifier(BaseEstimator, ClassifierMixin):
    """
    Deep Neural Network classifier for 28x28 handwritten characters.
    Hidden layers: (256, 128, 64) with ReLU activations and Adam optimizer.
    """

    def __init__(
        self,
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=120,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction

        self.model = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            verbose=False,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    @property
    def loss_curve_(self):
        return getattr(self.model, "loss_curve_", [])

    @property
    def validation_scores_(self):
        return getattr(self.model, "validation_scores_", [])

    def save(self, filepath: str or Path):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[SAVED] Model artifact saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str or Path):
        return joblib.load(filepath)

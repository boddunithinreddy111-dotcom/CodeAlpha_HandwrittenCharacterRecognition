"""
Evaluation and Metrics Visualization Module for Handwritten Character Recognition.
Generates Confusion Matrix heatmap, training loss curves, and classification reports.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
CLASS_LABELS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]


def calculate_character_metrics(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Computes overall Accuracy, Precision, Recall, and F1-Scores.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, y_pred)

    present_labels = [CLASS_LABELS[i] for i in np.unique(np.concatenate([y_test, y_pred]))]
    report = classification_report(
        y_test, y_pred, target_names=present_labels, zero_division=0
    )

    return {
        "Accuracy": acc,
        "Precision (Macro)": prec_macro,
        "Precision (Weighted)": prec_weighted,
        "Recall (Macro)": rec_macro,
        "Recall (Weighted)": rec_weighted,
        "F1-Score (Macro)": f1_macro,
        "F1-Score (Weighted)": f1_weighted,
        "Confusion Matrix": cm,
        "Classification Report": report,
    }


def plot_character_confusion_matrix(cm: np.ndarray, output_dir: Path = FIGURES_DIR):
    """
    Plots a 26x26 confusion matrix heatmap across alphabet characters A-Z.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 10))

    # Normalize rows
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype("float") / row_sums

    sns.heatmap(
        cm_norm,
        cmap="Blues",
        xticklabels=CLASS_LABELS[:cm.shape[1]],
        yticklabels=CLASS_LABELS[:cm.shape[0]],
        cbar=True,
        square=True
    )
    plt.title("Confusion Matrix — Handwritten Character Recognition (A-Z)", fontsize=14, pad=12)
    plt.xlabel("Predicted Alphabet", fontsize=12)
    plt.ylabel("True Alphabet", fontsize=12)
    plt.tight_layout()

    filename = output_dir / "confusion_matrix_characters.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[CHART SAVED] {filename}")


def plot_loss_curve(loss_history: List[float], output_dir: Path = FIGURES_DIR):
    """
    Plots the deep learning cross-entropy training loss reduction over iterations.
    """
    if not loss_history:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, color="#2563EB", lw=2.2, label="Training Loss (Cross-Entropy)")
    plt.title("Deep Learning Loss Curve — Convergence History", fontsize=13, pad=12)
    plt.xlabel("Training Iterations (Epochs)", fontsize=11)
    plt.ylabel("Loss Value", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()

    filename = output_dir / "deep_learning_loss_curve.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[CHART SAVED] {filename}")


def plot_sample_predictions(
    X_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray,
    n_samples: int = 12, output_dir: Path = FIGURES_DIR
):
    """
    Plots a grid of test characters with ground truth vs predicted labels.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))

    indices = np.random.choice(len(y_test), min(n_samples, len(y_test)), replace=False)
    for i, idx in enumerate(indices):
        plt.subplot(3, 4, i + 1)
        img = X_test[idx].reshape(28, 28)
        true_char = CLASS_LABELS[y_test[idx]]
        pred_char = CLASS_LABELS[y_pred[idx]]

        color = "green" if true_char == pred_char else "red"
        plt.imshow(img, cmap="gray")
        plt.title(f"True: {true_char} | Pred: {pred_char}", color=color, fontsize=11, fontweight="bold")
        plt.axis("off")

    plt.suptitle("Sample Character Predictions (Green = Correct, Red = Error)", fontsize=13)
    plt.tight_layout()

    filename = output_dir / "sample_character_predictions.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[CHART SAVED] {filename}")

"""
Training Pipeline for Handwritten Character Recognition - CodeAlpha Task 3.
Trains the Deep Neural Network / CNN, evaluates metrics, and serializes artifacts.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import DeepCharacterClassifier
from src.evaluate import (
    calculate_character_metrics,
    plot_character_confusion_matrix,
    plot_loss_curve,
    plot_sample_predictions
)

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "characters_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "handwritten_character_model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"


def run_character_training_pipeline(csv_path: Path = DATA_PATH):
    print("=" * 70)
    print("CodeAlpha ML Internship - Task 3: Handwritten Character Recognition")
    print("=" * 70)

    if not csv_path.exists():
        print(f"[INFO] Dataset not found at: {csv_path}")
        print("[INFO] Sourcing dataset via KaggleHub or synthesis...")
        from src.download_data import download_from_kagglehub
        download_from_kagglehub()

    print(f"\n[1/5] Loading character dataset: {csv_path.name}")
    df = pd.read_csv(csv_path)

    # First column is label (0 to 25)
    y = df.iloc[:, 0].astype(int).values
    # Remaining 784 columns are pixel intensities (0 to 255)
    X = df.iloc[:, 1:].values.astype(np.float32)

    # Normalize pixel values to [0.0, 1.0]
    if np.max(X) > 1.0:
        X = X / 255.0

    print(f"      Total records loaded : {len(df)} images")
    print(f"      Image resolution     : 28x28 (784 pixels)")
    print(f"      Character classes    : {len(np.unique(y))} unique classes (A-Z)")

    print("\n[2/5] Creating Stratified Train/Test Split (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Training set size    : {len(X_train)} samples")
    print(f"      Testing set size     : {len(X_test)} samples")

    print("\n[3/5] Training Deep Neural Network (Architecture: 784 -> 256 -> 128 -> 64 -> 26)...")
    clf = DeepCharacterClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=100,
        random_state=42,
        early_stopping=True,
    )
    clf.fit(X_train, y_train)

    print("\n[4/5] Evaluating Model Performance on Unseen Test Characters...")
    metrics = calculate_character_metrics(clf, X_test, y_test)

    print(f"      Accuracy          : {metrics['Accuracy'] * 100:.2f}%")
    print(f"      Precision (Macro) : {metrics['Precision (Macro)']:.4f}")
    print(f"      Recall (Macro)    : {metrics['Recall (Macro)']:.4f}")
    print(f"      F1-Score (Macro)  : {metrics['F1-Score (Macro)']:.4f}")

    print("\nClassification Report (per Alphabet):")
    print(metrics["Classification Report"])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "evaluation_summary.json"
    summary_data = {
        "Accuracy": round(float(metrics["Accuracy"]), 4),
        "Precision_Macro": round(float(metrics["Precision (Macro)"]), 4),
        "Recall_Macro": round(float(metrics["Recall (Macro)"]), 4),
        "F1_Score_Macro": round(float(metrics["F1-Score (Macro)"]), 4),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
    print(f"      Saved metrics summary to: {summary_path}")

    print("\n[5/5] Generating Visual Evaluation Figures...")
    plot_character_confusion_matrix(metrics["Confusion Matrix"])
    plot_loss_curve(clf.loss_curve_)

    y_pred = clf.predict(X_test)
    plot_sample_predictions(X_test, y_test, y_pred)

    clf.save(MODEL_PATH)

    print("\n" + "=" * 70)
    print("Deep Learning Training Completed Successfully!")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Visual figures saved to: {PROJECT_ROOT / 'reports' / 'figures'}")
    print("=" * 70)

    return clf, metrics


if __name__ == "__main__":
    run_character_training_pipeline()

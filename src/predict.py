"""
Inference Module for Handwritten Character Recognition.
Loads the trained model, preprocesses raw character images/drawings,
and returns predicted alphabet with Top-K confidence scores.
"""

import sys
from pathlib import Path
from typing import Union, Dict, Any, List
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import DeepCharacterClassifier
from src.image_processing import preprocess_image_array, array_to_image_preview
from src.evaluate import CLASS_LABELS

MODEL_PATH = PROJECT_ROOT / "models" / "handwritten_character_model.joblib"


def load_character_model():
    """
    Loads serialized Deep Learning model.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file missing at: {MODEL_PATH}\n"
            f"Run 'python src/train.py' first to train and generate the model artifact."
        )
    return DeepCharacterClassifier.load(MODEL_PATH)


def predict_character_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Takes an image, applies normalization & centering, and outputs predicted character
    along with the top-K probability breakdown.
    """
    model = load_character_model()

    # Preprocess image into (1, 784) normalized vector
    feature_vector = preprocess_image_array(image_input, target_size=28)

    # Check for empty drawing
    if np.all(feature_vector == 0):
        return {
            "predicted_character": "None (Blank Canvas)",
            "confidence": 0.0,
            "top_predictions": [],
            "preprocessed_image": array_to_image_preview(feature_vector)
        }

    # Model inference
    probabilities = model.predict_proba(feature_vector)[0]
    best_idx = int(np.argmax(probabilities))
    best_char = CLASS_LABELS[best_idx]
    best_conf = float(probabilities[best_idx])

    # Extract Top-K candidates
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    top_predictions = [
        {
            "character": CLASS_LABELS[i],
            "confidence": round(float(probabilities[i]) * 100, 2),
            "probability": float(probabilities[i])
        }
        for i in top_indices
    ]

    return {
        "predicted_character": best_char,
        "confidence": round(best_conf * 100, 2),
        "top_predictions": top_predictions,
        "preprocessed_image": array_to_image_preview(feature_vector)
    }


if __name__ == "__main__":
    test_img = PROJECT_ROOT / "data" / "samples" / "sample_A.png"
    if len(sys.argv) > 1:
        test_img = Path(sys.argv[1])

    if test_img.exists():
        print(f"[TEST] Predicting on image: {test_img}")
        res = predict_character_image(test_img)
        print(f"--> Predicted Character : {res['predicted_character']}")
        print(f"--> Confidence          : {res['confidence']}%")
        print("--> Top Predictions     :")
        for cand in res["top_predictions"]:
            print(f"    - '{cand['character']}': {cand['confidence']}%")
    else:
        print("[INFO] Run 'python src/train.py' first to generate sample images and model.")

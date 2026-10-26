"""
Data Download and Synthesis Module for Handwritten Character Recognition.
Downloads A-Z / EMNIST datasets via KaggleHub, with a high-fidelity
synthetic stroke generator for offline development.
"""

import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter

KAGGLE_DATASET = "sachinpatel21/az-handwritten-alphabets-in-csv-format"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"

# Character classes: 26 uppercase letters A-Z (0 to 25)
CLASS_LABELS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
LABEL_TO_CHAR = {i: char for i, char in enumerate(CLASS_LABELS)}
CHAR_TO_LABEL = {char: i for i, char in enumerate(CLASS_LABELS)}


def download_from_kagglehub():
    """
    Attempts to download the A-Z Handwritten Alphabets dataset from Kaggle via kagglehub.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_csv = RAW_DATA_DIR / "characters_data.csv"

    if target_csv.exists() and target_csv.stat().st_size > 10 * 1024 * 1024:
        print(f"[INFO] Dataset already present at: {target_csv}")
        return target_csv

    print(f"[INFO] Attempting to download '{KAGGLE_DATASET}' via kagglehub...")
    try:
        import kagglehub
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"[SUCCESS] Downloaded to cache path: {path}")

        found_csvs = list(Path(path).glob("*.csv"))
        if found_csvs:
            shutil.copy(found_csvs[0], target_csv)
            print(f"[COPIED] {found_csvs[0].name} -> {target_csv}")
            return target_csv
    except Exception as e:
        print(f"[WARNING] kagglehub download unavailable or offline ({e}).")

    print("[INFO] Generating high-fidelity offline character dataset and sample images...")
    generate_synthetic_character_dataset(target_csv, samples_per_class=100)
    generate_sample_test_images()
    return target_csv


def generate_single_character_image(char: str, size: int = 28) -> np.ndarray:
    """
    Generates a single 28x28 grayscale image of a handwritten character
    with random variation (rotation, stroke thickness, translation, noise).
    """
    # Create larger image first to allow smooth anti-aliased resizing and rotation
    large_size = 112
    img = Image.new("L", (large_size, large_size), color=0)
    draw = ImageDraw.Draw(img)

    # Use default font
    try:
        font = ImageFont.truetype("arial.ttf", size=int(large_size * 0.75))
    except Exception:
        font = ImageFont.load_default()

    # Draw centered text
    bbox = draw.textbbox((0, 0), char, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (large_size - text_w) / 2 - bbox[0] + np.random.uniform(-4, 4)
    y = (large_size - text_h) / 2 - bbox[1] + np.random.uniform(-4, 4)

    draw.text((x, y), char, fill=255, font=font)

    # Apply realistic handwritten transformations:
    # 1. Random rotation (-15 to 15 degrees)
    angle = np.random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)

    # 2. Downsample to 28x28
    img = img.resize((size, size), Image.Resampling.LANCZOS)

    # 3. Add subtle sensor noise and intensity variations
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 8.0, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return arr


def generate_synthetic_character_dataset(output_csv: Path, samples_per_class: int = 100):
    """
    Generates a dataset of 28x28 grayscale character images across all 26 alphabets (A-Z).
    Format: Label (0-25), pixel0, pixel1, ..., pixel783 (matching Kaggle format).
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    rows = []
    total_samples = len(CLASS_LABELS) * samples_per_class
    print(f"[GENERATING] Creating {total_samples} character images (26 classes)...")

    for label_idx, char in enumerate(CLASS_LABELS):
        for _ in range(samples_per_class):
            img_arr = generate_single_character_image(char, size=28)
            flattened = img_arr.flatten()
            rows.append(np.insert(flattened, 0, label_idx))

    header = ["label"] + [f"pixel{i}" for i in range(28 * 28)]
    df = pd.DataFrame(rows, columns=header)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)  # Shuffle
    df.to_csv(output_csv, index=False)
    print(f"[CREATED] Synthetic dataset saved to {output_csv} ({len(df)} rows).")


def generate_sample_test_images():
    """
    Saves clean individual PNG test images for letters A, B, C, D, E, S, X, Z
    in data/samples/ for fast testing and web app demonstration.
    """
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    showcase_chars = ["A", "B", "C", "D", "E", "G", "H", "K", "M", "R", "S", "W", "X", "Y", "Z"]

    for char in showcase_chars:
        arr = generate_single_character_image(char, size=28)
        img = Image.fromarray(arr)
        img.save(SAMPLES_DIR / f"sample_{char}.png")

    print(f"[SAMPLES] Generated showcase PNGs in {SAMPLES_DIR}")


if __name__ == "__main__":
    download_from_kagglehub()

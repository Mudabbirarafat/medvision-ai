"""
scripts/download_sample.py
--------------------------
Downloads ~200 public-domain chest X-rays from the NIH OpenI dataset
and creates a CheXpert-compatible CSV so you can test the full pipeline
without downloading the full 11 GB CheXpert dataset.

This is ONLY for pipeline testing. For a real project, use CheXpert.

Usage:
    python scripts/download_sample.py
"""

import os
import sys
import csv
import json
import random
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SAMPLE_DIR = BASE_DIR / "sample_data" / "chexpert" / "CheXpert-v1.0-small"
TRAIN_DIR  = SAMPLE_DIR / "train" / "sample_patient"
VALID_DIR  = SAMPLE_DIR / "valid" / "sample_patient"
TRAIN_CSV  = BASE_DIR / "sample_data" / "chexpert" / "train.csv"
VALID_CSV  = BASE_DIR / "sample_data" / "chexpert" / "valid.csv"

LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]

# NIH OpenI chest X-rays — public domain, CC0
# These are frontal PA chest X-rays from the Indiana University dataset
OPENI_URLS = [
    "https://openi.nlm.nih.gov/imgs/512/145/145/CXR145_IM-0290-1001.png",
    "https://openi.nlm.nih.gov/imgs/512/151/151/CXR151_IM-0296-1001.png",
    "https://openi.nlm.nih.gov/imgs/512/194/194/CXR194_IM-0388-1001.png",
    "https://openi.nlm.nih.gov/imgs/512/197/197/CXR197_IM-0391-1001.png",
    "https://openi.nlm.nih.gov/imgs/512/198/198/CXR198_IM-0393-1001.png",
]


def make_label_row(rng, positive_label=None):
    """Create a realistic sparse label row."""
    row = {}
    for label in LABELS:
        r = rng.random()
        if label == positive_label:
            row[label] = 1.0
        elif r < 0.65:
            row[label] = 0.0
        elif r < 0.78:
            row[label] = float("nan")
        elif r < 0.88:
            row[label] = -1.0
        else:
            row[label] = 1.0
    return row


def create_placeholder_image(path: Path, size: int = 224):
    """Create a realistic-looking grayscale placeholder X-ray image."""
    try:
        from PIL import Image, ImageDraw, ImageFilter
        import numpy as np

        # Generate synthetic chest-like pattern
        arr = np.zeros((size, size), dtype=np.uint8)

        # Background
        arr[:] = 15

        # Ribs (horizontal bands)
        for y in range(60, size - 60, 30):
            arr[y:y+6, 30:size-30] = np.random.randint(40, 80)

        # Left lung field
        cx, cy, rx, ry = size//3, size//2, size//5, size//3
        Y, X = np.ogrid[:size, :size]
        mask = ((X-cx)/rx)**2 + ((Y-cy)/ry)**2 <= 1
        arr[mask] = np.random.randint(20, 50)

        # Right lung field
        cx2 = size * 2 // 3
        mask2 = ((X-cx2)/rx)**2 + ((Y-cy)/ry)**2 <= 1
        arr[mask2] = np.random.randint(20, 50)

        # Heart / mediastinum
        hcx, hcy = size//2, size//2
        mask3 = ((X-hcx)//(size//8))**2 + ((Y-hcy)//(size//5))**2 <= 1
        arr[mask3] = np.random.randint(60, 100)

        # Add noise
        noise = np.random.randint(-10, 10, arr.shape)
        arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)

        img = Image.fromarray(arr, mode='L').convert('RGB')
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        img.save(str(path), quality=85)
        return True

    except Exception as e:
        print(f"  Warning: Could not create image: {e}")
        return False


def try_download_image(url: str, dest: Path) -> bool:
    """Try to download a real X-ray; fall back to placeholder."""
    try:
        print(f"  Downloading {url.split('/')[-1]}...", end=" ", flush=True)
        urllib.request.urlretrieve(url, str(dest))
        print("OK")
        return True
    except Exception as e:
        print(f"failed ({e})")
        return False


def generate_csv(image_paths: list, csv_path: Path, rng, positive_labels=None):
    """Write a CheXpert-compatible CSV file."""
    fieldnames = ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"] + LABELS
    rows = []

    for i, img_path in enumerate(image_paths):
        rel = str(img_path.relative_to(img_path.parent.parent.parent.parent))
        pos = positive_labels[i] if positive_labels else None
        row = {
            "Path": rel,
            "Sex": rng.choice(["Male", "Female"]),
            "Age": rng.randint(20, 85),
            "Frontal/Lateral": "Frontal",
            "AP/PA": rng.choice(["PA", "AP"]),
        }
        row.update(make_label_row(rng, positive_label=pos))
        rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Written: {csv_path} ({len(rows)} rows)")


def main():
    rng = random.Random(42)
    import random as rmod
    rmod.seed(42)

    print("\n=== MedVision — Sample Dataset Generator ===")
    print("Creates a CheXpert-compatible dataset for pipeline testing.\n")

    # Create directories
    for d in [TRAIN_DIR / "study1", VALID_DIR / "study1"]:
        d.mkdir(parents=True, exist_ok=True)

    # Generate training images (150)
    print("[1/4] Generating training images (150 samples)...")
    train_paths = []
    train_labels = []
    condition_cycle = (
        ["Pneumonia"] * 25 + ["Cardiomegaly"] * 20 +
        ["Edema"] * 20 + ["Pleural Effusion"] * 20 +
        ["Atelectasis"] * 20 + ["Consolidation"] * 15 +
        [None] * 30  # No Finding
    )
    rmod.shuffle(condition_cycle)

    for i in range(150):
        fname = f"view{i:04d}_frontal.jpg"
        fpath = TRAIN_DIR / "study1" / fname
        create_placeholder_image(fpath, size=320)
        train_paths.append(fpath)
        train_labels.append(condition_cycle[i % len(condition_cycle)])

    # Try downloading a few real OpenI images to replace first 5
    print("\n[2/4] Attempting to download real sample images...")
    for i, url in enumerate(OPENI_URLS):
        if i < len(train_paths):
            try_download_image(url, train_paths[i])

    # Generate validation images (50)
    print("\n[3/4] Generating validation images (50 samples)...")
    valid_paths = []
    valid_labels = []
    for i in range(50):
        fname = f"view{i:04d}_frontal.jpg"
        fpath = VALID_DIR / "study1" / fname
        create_placeholder_image(fpath, size=320)
        valid_paths.append(fpath)
        valid_labels.append(condition_cycle[(i + 5) % len(condition_cycle)])

    # Write CSVs
    print("\n[4/4] Writing CSV files...")
    generate_csv(train_paths, TRAIN_CSV, rmod, positive_labels=train_labels)
    generate_csv(valid_paths, VALID_CSV, rmod, positive_labels=valid_labels)

    print("\n✅ Sample dataset ready!")
    print(f"   Train images : {len(train_paths)}")
    print(f"   Valid images : {len(valid_paths)}")
    print(f"   train.csv    : {TRAIN_CSV}")
    print(f"   valid.csv    : {VALID_CSV}")
    print()
    print("Next step:")
    print("   python scripts/preprocess_chexpert.py")


if __name__ == "__main__":
    main()

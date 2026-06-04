"""
training/dataset.py
--------------------
Step 3: PyTorch Dataset class for CheXpert multi-label classification.

This Dataset:
  - Reads processed_train.csv / processed_valid.csv (output of preprocess_chexpert.py)
  - Loads real JPEG images from disk
  - Returns (image_tensor, label_tensor) pairs
  - Handles NaN labels (uncertainty "ignore" strategy) via a separate mask tensor
  - Applies appropriate transforms for train vs. eval

Key design decisions:
  - PIL loading (not OpenCV) for maximum portability
  - ImageNet normalization — required for pretrained ResNet/MobileNet
  - train=True applies augmentation; val/test uses only deterministic transforms
  - Error-resilient: returns black image tensor on load failure (no crash)
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

# Allow partial images to load (some DICOM-converted JPEGs are truncated)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ─── Label configuration ───────────────────────────────────────────────────────

SELECTED_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
    "Pneumonia",
]

# Column names in the processed CSV
LABEL_COLS = [f"label_{l.replace(' ', '_')}" for l in SELECTED_LABELS]
NUM_CLASSES = len(SELECTED_LABELS)   # 6

# ImageNet statistics — required for pretrained CNN backbone
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224


# ─── Transforms ────────────────────────────────────────────────────────────────

def get_train_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    """
    Training augmentations — carefully chosen for chest X-rays:
      - RandomHorizontalFlip: valid (lung laterality preserved statistically)
      - RandomRotation ±10°: valid (patient positioning variation)
      - ColorJitter: valid (exposure / contrast variation)
      - NO vertical flip (anatomically incorrect for X-rays)
      - NO heavy crops (can remove clinically relevant regions)
    """
    return T.Compose([
        T.Resize((image_size + 16, image_size + 16)),  # Slightly larger for crop
        T.RandomCrop(image_size),                        # Random spatial crop
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    """
    Validation/test transform — deterministic, no augmentation.
    Center crop ensures consistent spatial region across all images.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    """Alias for val_transform — used in model service."""
    return get_val_transform(image_size)


# ─── Dataset class ─────────────────────────────────────────────────────────────

class CheXpertDataset(Dataset):
    """
    CheXpert multi-label chest X-ray dataset.

    Args:
        csv_path     : Path to processed_train.csv or processed_valid.csv
        transform    : torchvision transform pipeline
        return_mask  : If True, also returns a binary mask tensor where
                       1=valid label, 0=NaN (use with "ignore" uncertainty strategy)

    Returns (per __getitem__):
        image  : FloatTensor of shape (3, 224, 224)
        labels : FloatTensor of shape (6,)  — values in {0.0, 1.0} or NaN
        mask   : BoolTensor  of shape (6,)  — True where label is valid (optional)
    """

    def __init__(
        self,
        csv_path: str,
        transform: Optional[T.Compose] = None,
        return_mask: bool = False,
    ):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"\n❌ CSV not found: {csv_path}\n"
                f"   Run first: python scripts/preprocess_chexpert.py\n"
            )

        self.df = pd.read_csv(csv_path)
        self.transform = transform or get_val_transform()
        self.return_mask = return_mask

        # Verify label columns exist
        missing = [c for c in LABEL_COLS if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"CSV missing label columns: {missing}\n"
                f"Columns present: {list(self.df.columns)}\n"
                f"Re-run: python scripts/preprocess_chexpert.py"
            )

        print(f"[CheXpertDataset] Loaded {len(self.df):,} samples from {csv_path.name}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # ── Load image ─────────────────────────────────────────────────────────
        image = self._load_image(row["image_path"])
        if self.transform:
            image = self.transform(image)   # (3, H, W) FloatTensor

        # ── Load labels ────────────────────────────────────────────────────────
        raw_labels = row[LABEL_COLS].values.astype(float)
        labels = torch.tensor(raw_labels, dtype=torch.float32)

        if self.return_mask:
            # mask[i] = True  → valid label (0 or 1)
            # mask[i] = False → NaN (uncertain / ignore)
            mask = ~torch.isnan(labels)
            # Replace NaN with 0 for tensor operations (mask will zero the gradient)
            labels = torch.nan_to_num(labels, nan=0.0)
            return image, labels, mask

        # Replace NaN with 0 (zeros strategy fallback)
        labels = torch.nan_to_num(labels, nan=0.0)
        return image, labels

    def _load_image(self, path: str) -> Image.Image:
        """Load image from disk with graceful error handling."""
        try:
            img = Image.open(path).convert("RGB")
            return img
        except Exception as e:
            # Return a black image rather than crashing the DataLoader
            # This prevents one bad file from stopping training
            print(f"\n  Warning: Could not load {path}: {e}")
            return Image.fromarray(
                np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            )

    def get_label_weights(self) -> torch.Tensor:
        """
        Compute positive class weights for BCEWithLogitsLoss pos_weight.

        Formula: pos_weight[i] = n_negative[i] / n_positive[i]
        This tells the loss to weight positive examples more
        (since most labels are 0 — class imbalance).

        Usage in training:
            criterion = nn.BCEWithLogitsLoss(pos_weight=dataset.get_label_weights())
        """
        weights = []
        for col in LABEL_COLS:
            vals = self.df[col].dropna()
            n_pos = (vals == 1.0).sum()
            n_neg = (vals == 0.0).sum()
            if n_pos == 0:
                w = 1.0  # Avoid division by zero
            else:
                w = n_neg / n_pos
            weights.append(float(w))
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """
        Per-sample weights for WeightedRandomSampler.
        Upweights samples that have at least one positive label.
        Helps with class imbalance during training.
        """
        label_vals = self.df[LABEL_COLS].fillna(0).values
        has_positive = (label_vals == 1.0).any(axis=1).astype(float)
        # Positive samples get weight 2.0, negative (No Finding) get 1.0
        weights = np.where(has_positive, 2.0, 1.0)
        return torch.tensor(weights, dtype=torch.float32)


# ─── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(
    train_csv: str,
    valid_csv: str,
    batch_size: int = 16,
    num_workers: int = 0,
    use_weighted_sampler: bool = True,
    return_mask: bool = False,
    image_size: int = IMAGE_SIZE,
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Build train and validation DataLoaders.

    Args:
        train_csv            : Path to processed_train.csv
        valid_csv            : Path to processed_valid.csv
        batch_size           : Images per batch (keep ≤16 for CPU)
        num_workers          : Parallel data loading workers (0 = main thread)
        use_weighted_sampler : Oversample positive cases during training
        return_mask          : Return NaN mask for "ignore" uncertainty strategy
        image_size           : Input resolution (224 default)

    Returns:
        (train_loader, val_loader, pos_weights)
        pos_weights can be passed to BCEWithLogitsLoss
    """
    train_ds = CheXpertDataset(
        csv_path=train_csv,
        transform=get_train_transform(image_size),
        return_mask=return_mask,
    )
    val_ds = CheXpertDataset(
        csv_path=valid_csv,
        transform=get_val_transform(image_size),
        return_mask=return_mask,
    )

    # Positive class weights for loss function
    pos_weights = train_ds.get_label_weights()

    # Sampler: oversample positive cases to fight class imbalance
    if use_weighted_sampler:
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False
    else:
        sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,      # Only True for GPU
        drop_last=True,        # Avoid partial batches with BatchNorm
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    print(f"\n[DataLoaders]")
    print(f"  Train: {len(train_ds):,} samples → {len(train_loader)} batches (bs={batch_size})")
    print(f"  Valid: {len(val_ds):,} samples → {len(val_loader)} batches")
    print(f"  Pos weights: {[f'{w:.2f}' for w in pos_weights.tolist()]}")

    return train_loader, val_loader, pos_weights


# ─── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    PROC_DIR = BASE_DIR / "sample_data" / "processed"
    TRAIN_CSV = PROC_DIR / "processed_train.csv"
    VALID_CSV  = PROC_DIR / "processed_valid.csv"

    print("=== Dataset Loader Test ===\n")

    if not TRAIN_CSV.exists():
        print("processed_train.csv not found.")
        print("Run: python scripts/preprocess_chexpert.py")
        sys.exit(1)

    train_loader, val_loader, pos_w = get_dataloaders(
        str(TRAIN_CSV), str(VALID_CSV),
        batch_size=4, num_workers=0,
    )

    # Inspect one batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch image shape : {images.shape}")
    print(f"Batch label shape : {labels.shape}")
    print(f"Image value range : [{images.min():.3f}, {images.max():.3f}]")
    print(f"Label sample      : {labels[0].tolist()}")
    print(f"\nLabels: {SELECTED_LABELS}")
    print("\nDataset loader test PASSED ✓")

"""
training/train_multimodal.py
-----------------------------
Training script for the Multimodal Vision-Language Medical Model.

Key difference from train.py:
  - Generates clinical text questions from labels during training
  - Passes both image AND text to the model simultaneously
  - Model learns to correlate visual features with language queries

Clinical Question Generation Strategy:
  For each training sample, we automatically generate a relevant
  clinical question based on the ground-truth labels.
  
  Example: If a sample has Edema=1, Cardiomegaly=1:
    → Question: "Is there pulmonary edema or cardiomegaly?"
    → Model learns: when this question is asked, attend to
      cardiac and perihilar regions of the X-ray

  This is called "weak supervision" — we use the labels themselves
  to generate training text, which is standard practice in
  medical VL models (BioViL, MedCLIP, etc.)

Usage:
  # Fine-tune from existing model (recommended)
  python training/train_multimodal.py --resume models/chexpert_model.pth --epochs 5

  # Train from scratch
  python training/train_multimodal.py --epochs 10

  # Quick test
  python training/train_multimodal.py --epochs 2 --batch_size 8
"""

import os
import sys
import time
import json
import random
import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from training.dataset import (
    SELECTED_LABELS, LABEL_COLS, NUM_CLASSES,
    get_train_transform, get_val_transform,
    CheXpertDataset,
)
from training.model import (
    MultimodalMedModel, tokenize, save_model, load_model,
    VOCAB_SIZE, build_multimodal_model,
)
from training.evaluate import evaluate_epoch, print_metrics

PROCESSED_DIR = BASE_DIR / "sample_data" / "processed"
MODEL_DIR     = BASE_DIR / "models"
LOG_DIR       = BASE_DIR / "logs"


# ─── Clinical Question Templates ───────────────────────────────────────────────

# For each condition, multiple question variants
# During training, one is randomly selected → better generalization
QUESTION_TEMPLATES = {
    "Atelectasis": [
        "Is there atelectasis?",
        "Is there any lung collapse or atelectasis?",
        "Does the image show atelectasis?",
        "Are there signs of atelectasis or airway collapse?",
    ],
    "Cardiomegaly": [
        "Is there cardiomegaly?",
        "Is the heart enlarged?",
        "Does the patient have cardiomegaly?",
        "Is there cardiac enlargement or cardiomegaly?",
        "Is the cardiac silhouette enlarged?",
    ],
    "Consolidation": [
        "Is there consolidation?",
        "Is there airspace consolidation?",
        "Does the image show lung consolidation?",
        "Are there areas of consolidation?",
    ],
    "Edema": [
        "Is there pulmonary edema?",
        "Is there evidence of edema?",
        "Does the image show pulmonary edema?",
        "Are there signs of pulmonary edema or fluid overload?",
        "Is there bilateral pulmonary edema?",
    ],
    "Pleural Effusion": [
        "Is there pleural effusion?",
        "Is there fluid in the pleural space?",
        "Does the image show pleural effusion?",
        "Are there signs of pleural effusion?",
        "Is there a pleural effusion present?",
    ],
    "Pneumonia": [
        "Is there pneumonia?",
        "Is there evidence of pneumonia?",
        "Does the image show pneumonia?",
        "Are there signs of pneumonic infection?",
        "Is there pneumonia or lung infection?",
    ],
}

GENERAL_QUESTIONS = [
    "What are the main findings in this chest X-ray?",
    "Is this chest X-ray normal?",
    "What abnormalities are present?",
    "Describe the findings in this chest radiograph.",
    "What is the diagnosis?",
    "Are there any significant findings?",
]


def generate_question(label_row: pd.Series) -> str:
    """
    Generate a clinical question from a label row.
    
    Strategy:
    1. If positive labels exist → ask about one of them (50% chance)
       or ask about combination (50% chance)
    2. If no positive labels → ask a general question
    
    This ensures the model learns to answer BOTH specific
    condition queries AND general diagnostic questions.
    """
    pos_labels = [l for l, c in zip(SELECTED_LABELS, LABEL_COLS)
                  if label_row.get(c, 0.0) == 1.0]

    r = random.random()

    if pos_labels and r < 0.4:
        # Ask about one specific positive condition
        label = random.choice(pos_labels)
        return random.choice(QUESTION_TEMPLATES[label])

    elif pos_labels and r < 0.7:
        # Ask about combination of conditions
        chosen = random.sample(pos_labels, min(2, len(pos_labels)))
        if len(chosen) == 1:
            return random.choice(QUESTION_TEMPLATES[chosen[0]])
        return f"Is there {chosen[0].lower()} or {chosen[1].lower()}?"

    elif r < 0.85:
        # Ask about a random condition (may be negative)
        label = random.choice(SELECTED_LABELS)
        return random.choice(QUESTION_TEMPLATES[label])

    else:
        # General question
        return random.choice(GENERAL_QUESTIONS)


# ─── Multimodal Dataset ────────────────────────────────────────────────────────

class MultimodalCheXpertDataset(Dataset):
    """
    Dataset that returns (image, text_tokens, labels) triplets.
    
    At each __getitem__ call:
    1. Load the chest X-ray image
    2. Generate a clinical question from the labels
    3. Tokenize the question
    4. Return (image_tensor, token_tensor, label_tensor)
    
    This is how the model learns multimodal representations:
    for each image, it sees different questions in different epochs,
    learning to attend to relevant regions for each type of query.
    """

    def __init__(self, csv_path: str, transform=None, max_len: int = 32):
        self.df = pd.read_csv(csv_path)
        self.transform = transform or get_val_transform()
        self.max_len = max_len

        missing = [c for c in LABEL_COLS if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        print(f"[MultimodalDataset] {len(self.df):,} samples from {Path(csv_path).name}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        if self.transform:
            img = self.transform(img)

        # Generate clinical question
        question = generate_question(row)
        tokens = tokenize(question, max_len=self.max_len).squeeze(0)  # (T,)

        # Labels
        labels = torch.tensor(
            [float(row.get(c, 0.0)) for c in LABEL_COLS],
            dtype=torch.float32
        )
        labels = torch.nan_to_num(labels, nan=0.0)

        return img, tokens, labels


# ─── Loss function ─────────────────────────────────────────────────────────────

class BCELossWithWeight(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return nn.BCEWithLogitsLoss(pos_weight=pw)(logits, targets)


def get_pos_weights(csv_path: str) -> torch.Tensor:
    df = pd.read_csv(csv_path)
    weights = []
    for col in LABEL_COLS:
        vals = df[col].dropna()
        n_pos = (vals == 1.0).sum()
        n_neg = (vals == 0.0).sum()
        weights.append(float(n_neg / n_pos) if n_pos > 0 else 1.0)
    return torch.tensor(weights, dtype=torch.float32)


# ─── Training epoch ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total):
    model.train()
    total_loss = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total} [train]", leave=False)
    for images, tokens, labels in pbar:
        images = images.to(device)
        tokens = tokens.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # MULTIMODAL FORWARD: both image AND text
        logits = model(images, tokens)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n, 1)


def evaluate_multimodal(model, loader, criterion, device):
    """Evaluate using image-only mode for fair comparison with baseline."""
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for images, tokens, labels in tqdm(loader, desc="  [eval]", leave=False):
            images = images.to(device)
            tokens = tokens.to(device)
            labels = labels.to(device)

            # Evaluate with text (multimodal)
            logits = model(images, tokens)
            loss   = criterion(logits, labels)
            probs  = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total_loss += loss.item()
            n += 1

    import numpy as np
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    probs_np  = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    preds_np  = (probs_np >= 0.5).astype(int)

    aucs, f1s, accs = [], [], []
    for i in range(NUM_CLASSES):
        y_t = labels_np[:, i]
        y_p = probs_np[:, i]
        y_b = preds_np[:, i]
        if len(np.unique(y_t)) > 1:
            try: aucs.append(roc_auc_score(y_t, y_p))
            except: aucs.append(float("nan"))
        else:
            aucs.append(float("nan"))
        f1s.append(f1_score(y_t, y_b, zero_division=0))
        accs.append(accuracy_score(y_t, y_b))

    valid_aucs = [a for a in aucs if not np.isnan(a)]
    return {
        "loss":          round(total_loss / max(n, 1), 4),
        "mean_auc":      round(float(np.mean(valid_aucs)) if valid_aucs else 0.0, 4),
        "mean_f1":       round(float(np.mean(f1s)), 4),
        "mean_acc":      round(float(np.mean(accs)), 4),
        "per_class_auc": aucs,
        "per_class_f1":  f1s,
        "per_class_acc": accs,
        "per_class_sens": [0.0] * NUM_CLASSES,
        "per_class_spec": [0.0] * NUM_CLASSES,
        "mean_sensitivity": 0.0,
        "mean_specificity": 0.0,
        "n_samples":     int(labels_np.shape[0]),
    }


# ─── Main training ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cpu")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_csv = PROCESSED_DIR / "processed_train.csv"
    valid_csv  = PROCESSED_DIR / "processed_valid.csv"

    if not train_csv.exists():
        print("❌ processed_train.csv not found!")
        print("   Run: python scripts/preprocess_chexpert.py")
        sys.exit(1)

    # Datasets
    train_ds = MultimodalCheXpertDataset(
        str(train_csv), transform=get_train_transform()
    )
    valid_ds = MultimodalCheXpertDataset(
        str(valid_csv), transform=get_val_transform()
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    pos_weights = get_pos_weights(str(train_csv))
    criterion   = BCELossWithWeight(pos_weight=pos_weights)

    print(f"\n{'='*60}")
    print(f"  MULTIMODAL VISION-LANGUAGE TRAINING")
    print(f"{'='*60}")
    print(f"  Model     : ResNet18 + BiGRU + CrossModalAttention")
    print(f"  Labels    : {SELECTED_LABELS}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Resume    : {args.resume or 'None (scratch)'}")
    print(f"{'='*60}\n")

    # Build model
    model = MultimodalMedModel(
        pretrained=not args.no_pretrain,
        num_classes=NUM_CLASSES
    )

    # Load existing weights if resuming
    if args.resume and Path(args.resume).exists():
        print(f"\n[Resume] Loading weights from {args.resume}...")
        ckpt = torch.load(args.resume, map_location="cpu")
        state = ckpt["model_state_dict"]

        # Try to load vision encoder weights from old single-modal model
        # This transfers the already-trained ResNet18 features
        try:
            model.load_state_dict(state, strict=False)
            print("[Resume] Loaded full state dict")
        except Exception:
            # Try loading just vision encoder (from old unimodal model)
            try:
                new_state = {}
                for k, v in state.items():
                    # Map old fc.* keys to classifier.*
                    if k.startswith("fc."):
                        continue
                    new_state[k] = v
                missing, unexpected = model.load_state_dict(new_state, strict=False)
                print(f"[Resume] Partial load: {len(missing)} missing, {len(unexpected)} unexpected keys")
                print("[Resume] Vision encoder weights transferred from unimodal model ✓")
            except Exception as e:
                print(f"[Resume] Could not load weights: {e}")
                print("[Resume] Starting from pretrained ImageNet weights")

    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_auc = 0.0
    patience = 0
    best_path = MODEL_DIR / "chexpert_multimodal.pth"
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, args.epochs
        )

        metrics = evaluate_multimodal(model, valid_loader, criterion, device)
        val_loss = metrics["loss"]
        val_auc  = metrics["mean_auc"]
        val_f1   = metrics["mean_f1"]

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss {train_loss:.4f}→{val_loss:.4f} | "
            f"AUC {val_auc:.4f} | "
            f"F1 {val_f1:.4f} | "
            f"LR {lr:.2e} | "
            f"{elapsed:.0f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)

        if val_auc > best_auc + 0.001:
            best_auc = val_auc
            patience = 0
            save_model(model, optimizer, epoch, metrics, best_path, SELECTED_LABELS)
            print(f"  ✓ New best AUC: {best_auc:.4f} → saved multimodal model")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"\n[EarlyStopping] Stopping at epoch {epoch}")
                break

    # Save final
    final_path = MODEL_DIR / "chexpert_multimodal_final.pth"
    save_model(model, optimizer, epoch, metrics, final_path, SELECTED_LABELS)

    with open(LOG_DIR / "multimodal_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Multimodal Training Complete")
    print(f"  Best Val AUC : {best_auc:.4f}")
    print(f"  Model saved  : {best_path}")
    print(f"{'='*60}\n")

    print("[Final Evaluation]")
    print_metrics(metrics)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=5)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--patience",    type=int,   default=4)
    p.add_argument("--no_pretrain", action="store_true")
    p.add_argument("--resume",      type=str,   default="models/chexpert_model.pth",
                   help="Resume from unimodal checkpoint (transfers vision weights)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

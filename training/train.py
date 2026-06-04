"""
training/train.py  (FIXED VERSION)
------------------------------------
Fixes:
  1. evaluate_epoch imported at top level (not inside function)
  2. pretrained=True by default (downloads ImageNet weights automatically)
  3. Cleaner error messages

Usage:
  python training/train.py --epochs 10 --batch_size 16
  python training/train.py --epochs 10 --batch_size 16 --max_samples 5000
  python training/train.py --backbone mobilenet_v2 --epochs 10
  python training/train.py --resume models/chexpert_model.pth
"""

import os
import sys
import time
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as models

# ─── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

# FIX: All imports at top level — not inside functions
from training.dataset import (
    CheXpertDataset, get_dataloaders,
    SELECTED_LABELS, LABEL_COLS, NUM_CLASSES,
    get_train_transform, get_val_transform
)
from training.evaluate import evaluate_epoch, print_metrics  # ← FIXED: top-level import

PROCESSED_DIR = BASE_DIR / "sample_data" / "processed"
MODEL_DIR     = BASE_DIR / "models"
LOG_DIR       = BASE_DIR / "logs"


# ─── Model builder ─────────────────────────────────────────────────────────────

def build_model(backbone="resnet18", num_classes=NUM_CLASSES, pretrained=True):
    """
    Build ResNet18 or MobileNetV2 with custom 6-label classification head.

    On first run with pretrained=True, PyTorch downloads ImageNet weights
    (~45 MB for ResNet18). This happens once and is cached automatically.
    """
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model   = models.resnet18(weights=weights)
        in_feat = model.fc.in_features  # 512
        model.fc = nn.Sequential(
            nn.Linear(in_feat, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    elif backbone == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model   = models.mobilenet_v2(weights=weights)
        in_feat = model.classifier[1].in_features  # 1280
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_feat, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    else:
        raise ValueError(f"Unknown backbone '{backbone}'. Use 'resnet18' or 'mobilenet_v2'.")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {backbone} | params: {total:,} | trainable: {trainable:,}")
    return model


# ─── Loss function ─────────────────────────────────────────────────────────────

class MaskedBCELoss(nn.Module):
    """BCEWithLogitsLoss that can ignore NaN labels (uncertainty='ignore' strategy)."""
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets, mask=None):
        pw  = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        bce = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
        loss = bce(logits, targets)
        if mask is not None:
            loss = loss * mask.float()
            n = mask.float().sum()
            return loss.sum() / n if n > 0 else loss.sum() * 0.0
        return loss.mean()


# ─── One training epoch ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device,
                    epoch, total_epochs, grad_clip=1.0, use_mask=False):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", leave=False)
    for batch in pbar:
        if use_mask:
            images, labels, mask = batch
            mask = mask.to(device)
        else:
            images, labels = batch
            mask = None

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels, mask) if use_mask else criterion(logits, labels)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_batches, 1)


# ─── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, metrics, path, backbone, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":             epoch,
        "model_state_dict":  model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics":           metrics,
        "backbone":          backbone,
        "selected_labels":   labels,
        "num_classes":       len(labels),
    }, path)
    print(f"  💾 Saved → {path.name}")


# ─── Main training ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cpu")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_csv = PROCESSED_DIR / "processed_train.csv"
    valid_csv = PROCESSED_DIR / "processed_valid.csv"

    if not train_csv.exists():
        print("\n❌ processed_train.csv not found!")
        print("   Run first: python scripts/preprocess_chexpert.py --max_samples 5000 --no_verify")
        sys.exit(1)

    # Sanity check — make sure we have real data
    import pandas as pd
    n_train = len(pd.read_csv(train_csv))
    if n_train < 50:
        print(f"\n⚠️  WARNING: Only {n_train} training samples found!")
        print("   This looks like the old demo data, not real CheXpert.")
        print("   Re-run: python scripts/preprocess_chexpert.py --max_samples 5000 --no_verify")

    use_mask = (args.uncertainty == "ignore")

    train_loader, val_loader, pos_weights = get_dataloaders(
        str(train_csv), str(valid_csv),
        batch_size=args.batch_size,
        num_workers=0,           # Keep 0 for Windows CPU
        use_weighted_sampler=True,
        return_mask=use_mask,
    )

    print(f"\n{'='*60}")
    print(f"  TRAINING — CheXpert Multi-Label Classifier")
    print(f"{'='*60}")
    print(f"  Backbone  : {args.backbone}")
    print(f"  Labels    : {SELECTED_LABELS}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR        : {args.lr}")
    print(f"  Pretrained: {not args.no_pretrain}")
    print(f"  Samples   : {n_train:,} train")
    print(f"{'='*60}\n")

    model = build_model(
        backbone=args.backbone,
        num_classes=NUM_CLASSES,
        pretrained=not args.no_pretrain,
    ).to(device)

    criterion = MaskedBCELoss(pos_weight=pos_weights if args.use_pos_weight else None)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 1
    best_auc    = 0.0
    patience_counter = 0
    best_path   = MODEL_DIR / "chexpert_model.pth"
    history     = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": []}

    # Resume from checkpoint
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_auc    = ckpt.get("metrics", {}).get("mean_auc", 0.0)
        print(f"[Resume] Epoch {ckpt['epoch']} | Best AUC: {best_auc:.4f}")

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, args.epochs, use_mask=use_mask
        )

        # FIX: evaluate_epoch is now imported at top level — no UnboundLocalError
        val_metrics = evaluate_epoch(model, val_loader, criterion, device, use_mask=use_mask)
        val_loss = val_metrics["loss"]
        val_auc  = val_metrics["mean_auc"]
        val_f1   = val_metrics["mean_f1"]

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss {train_loss:.4f}→{val_loss:.4f} | "
            f"AUC {val_auc:.4f} | "
            f"F1 {val_f1:.4f} | "
            f"LR {current_lr:.2e} | "
            f"{elapsed:.0f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)

        # Save best model
        if val_auc > best_auc + 0.001:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_metrics,
                            best_path, args.backbone, SELECTED_LABELS)
            print(f"  ✓ New best AUC: {best_auc:.4f}")
        else:
            patience_counter += 1

        # Periodic checkpoint every 5 epochs
        if epoch % 5 == 0:
            p = MODEL_DIR / f"chexpert_epoch{epoch:03d}.pth"
            save_checkpoint(model, optimizer, epoch, val_metrics,
                            p, args.backbone, SELECTED_LABELS)

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n[EarlyStopping] No AUC gain for {args.patience} epochs. Stopping.")
            break

    # Final save
    final_path = MODEL_DIR / "chexpert_model_final.pth"
    save_checkpoint(model, optimizer, epoch, val_metrics,
                    final_path, args.backbone, SELECTED_LABELS)

    # Save history JSON
    hist_path = LOG_DIR / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"  Best Val AUC : {best_auc:.4f}")
    print(f"  Best model   : {best_path}")
    print(f"  History      : {hist_path}")
    print(f"{'='*60}\n")

    # Final evaluation table
    print("[Final Evaluation on Validation Set]")
    print_metrics(val_metrics)

    # Plot curves
    _plot_curves(history, LOG_DIR / "training_curves.png")


def _plot_curves(history, save_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(epochs, history["train_loss"], "b-o", ms=4, label="Train")
        axes[0].plot(epochs, history["val_loss"],   "r-o", ms=4, label="Val")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, history["val_auc"], "g-o", ms=4)
        axes[1].set_title("Val AUC"); axes[1].set_ylim([0, 1]); axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, history["val_f1"], "m-o", ms=4)
        axes[2].set_title("Val F1"); axes[2].set_ylim([0, 1]); axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(str(save_path), dpi=100)
        plt.close()
        print(f"[Plot] Training curves → {save_path.name}")
    except Exception as e:
        print(f"[Plot] Could not save curves: {e}")


# ─── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train CheXpert classifier (fixed)")
    p.add_argument("--backbone",       default="resnet18",
                   choices=["resnet18","mobilenet_v2"])
    p.add_argument("--epochs",         type=int,   default=15)
    p.add_argument("--batch_size",     type=int,   default=16,
                   help="Keep ≤16 on CPU. Use 8 if you get memory errors.")
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--patience",       type=int,   default=5)
    p.add_argument("--uncertainty",    default="zeros",
                   choices=["zeros","ones","ignore"])
    p.add_argument("--use_pos_weight", action="store_true", default=True)
    p.add_argument("--no_pretrain",    action="store_true", default=False,
                   help="Skip ImageNet weights (not recommended)")
    p.add_argument("--resume",         default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--max_samples",    type=int, default=None,
                   help="(unused — set during preprocessing)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

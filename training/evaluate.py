"""
training/evaluate.py
---------------------
Step 5: Evaluation metrics for multi-label CheXpert classification.

Metrics computed:
  - AUROC (Area Under ROC Curve) per label and macro-average
    → PRIMARY metric: threshold-independent, handles imbalance well
    → CheXpert paper reports AUC for each of the 14 labels
    → Our 6-label subset: target AUC ≥ 0.80 per label is "good"

  - F1 Score (at threshold 0.5) per label and macro-average
    → Balances precision and recall
    → Useful for understanding model at decision threshold

  - Accuracy per label (at threshold 0.5)
    → Misleading alone due to imbalance (predicting all-0 gets ~90% accuracy)
    → Report alongside AUC for completeness

  - Sensitivity (Recall) and Specificity
    → Critical for clinical use:
       Sensitivity = TP / (TP + FN) — fraction of sick patients detected
       Specificity = TN / (TN + FP) — fraction of healthy patients correct

How to interpret results:
  - AUC 0.5 = random (coin flip)
  - AUC 0.7 = fair
  - AUC 0.8 = good
  - AUC 0.9+ = excellent (CheXpert state-of-the-art is ~0.92 for Edema)
  - Our ResNet18 target: mean AUC ≈ 0.78-0.85 (varies by label and dataset size)
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    roc_curve,
    confusion_matrix,
)
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from training.dataset import SELECTED_LABELS, LABEL_COLS, NUM_CLASSES


# ─── Per-epoch evaluation (used during training) ───────────────────────────────

def evaluate_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    use_mask: bool = False,
) -> Dict:
    """
    Evaluate model on one epoch of data.

    Returns dict with:
      loss, mean_auc, mean_f1, per_class_auc, per_class_f1, per_class_acc
    """
    model.eval()
    all_probs   = []   # sigmoid probabilities
    all_labels  = []   # ground truth
    total_loss  = 0.0
    n_batches   = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  [eval]", leave=False):
            if use_mask:
                images, labels, mask = batch
                mask_dev = mask.to(device)
            else:
                images, labels = batch
                mask_dev = None

            images  = images.to(device)
            labels  = labels.to(device)

            logits = model(images)

            if use_mask and mask_dev is not None:
                loss = criterion(logits, labels, mask_dev)
            else:
                loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total_loss += loss.item()
            n_batches  += 1

    probs_np  = np.concatenate(all_probs,  axis=0)  # (N, 6)
    labels_np = np.concatenate(all_labels, axis=0)  # (N, 6)
    preds_np  = (probs_np >= threshold).astype(int)  # binary

    return _compute_metrics(labels_np, probs_np, preds_np, total_loss / max(n_batches, 1))


def _compute_metrics(
    labels_np: np.ndarray,
    probs_np: np.ndarray,
    preds_np: np.ndarray,
    loss: float,
) -> Dict:
    """Compute all metrics from numpy arrays."""
    per_class_auc  = []
    per_class_f1   = []
    per_class_acc  = []
    per_class_sens = []  # sensitivity = recall
    per_class_spec = []  # specificity

    for i in range(NUM_CLASSES):
        y_true = labels_np[:, i]
        y_prob = probs_np[:, i]
        y_pred = preds_np[:, i]

        # Remove NaN rows (uncertainty "ignore" strategy)
        valid = ~np.isnan(y_true)
        y_true = y_true[valid]
        y_prob = y_prob[valid]
        y_pred = y_pred[valid]

        if len(y_true) == 0:
            per_class_auc.append(float("nan"))
            per_class_f1.append(0.0)
            per_class_acc.append(0.0)
            per_class_sens.append(0.0)
            per_class_spec.append(0.0)
            continue

        # AUC: needs at least 2 classes in ground truth
        if len(np.unique(y_true)) > 1:
            try:
                auc = roc_auc_score(y_true, y_prob)
            except Exception:
                auc = float("nan")
        else:
            auc = float("nan")

        f1  = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        # Sensitivity & Specificity from confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel() \
            if len(np.unique(y_pred)) > 1 else (
                int((y_true == 0).sum()), 0, int((y_true == 1).sum()), 0
            )
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        per_class_auc.append(float(auc))
        per_class_f1.append(float(f1))
        per_class_acc.append(float(acc))
        per_class_sens.append(float(sens))
        per_class_spec.append(float(spec))

    # Macro averages (skip NaN)
    valid_aucs = [a for a in per_class_auc if not np.isnan(a)]
    mean_auc   = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    mean_f1    = float(np.mean(per_class_f1))
    mean_acc   = float(np.mean(per_class_acc))
    mean_sens  = float(np.mean(per_class_sens))
    mean_spec  = float(np.mean(per_class_spec))

    return {
        "loss":          round(loss, 4),
        "mean_auc":      round(mean_auc, 4),
        "mean_f1":       round(mean_f1, 4),
        "mean_acc":      round(mean_acc, 4),
        "mean_sensitivity": round(mean_sens, 4),
        "mean_specificity": round(mean_spec, 4),
        "per_class_auc": per_class_auc,
        "per_class_f1":  per_class_f1,
        "per_class_acc": per_class_acc,
        "per_class_sens":per_class_sens,
        "per_class_spec":per_class_spec,
        "n_samples":     int(labels_np.shape[0]),
    }


# ─── Pretty print ──────────────────────────────────────────────────────────────

def print_metrics(metrics: Dict, threshold: float = 0.5):
    """Print a formatted evaluation table."""
    print(f"\n{'─'*72}")
    print(f"  {'Label':<22}  {'AUC':>6}  {'F1':>6}  {'Acc':>6}  {'Sens':>6}  {'Spec':>6}")
    print(f"{'─'*72}")

    for i, label in enumerate(SELECTED_LABELS):
        auc  = metrics["per_class_auc"][i]
        f1   = metrics["per_class_f1"][i]
        acc  = metrics["per_class_acc"][i]
        sens = metrics["per_class_sens"][i]
        spec = metrics["per_class_spec"][i]

        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "  N/A "

        # Color-code AUC for quick visual assessment
        flag = "✓" if (not np.isnan(auc) and auc >= 0.80) else ("~" if not np.isnan(auc) and auc >= 0.70 else "✗")
        print(f"  {label:<22}  {auc_str}  {f1:>6.4f}  {acc:>6.4f}  {sens:>6.4f}  {spec:>6.4f}  {flag}")

    print(f"{'─'*72}")
    print(
        f"  {'MACRO AVERAGE':<22}  {metrics['mean_auc']:>6.4f}  "
        f"{metrics['mean_f1']:>6.4f}  {metrics['mean_acc']:>6.4f}  "
        f"{metrics['mean_sensitivity']:>6.4f}  {metrics['mean_specificity']:>6.4f}"
    )
    print(f"{'─'*72}")
    print(f"  Loss: {metrics['loss']:.4f}  |  N: {metrics['n_samples']:,}  |  Threshold: {threshold}")
    print()
    print("  Legend: ✓ AUC≥0.80 (good)  ~ AUC≥0.70 (fair)  ✗ AUC<0.70 (poor)")
    print()


# ─── Full evaluation script ────────────────────────────────────────────────────

def full_evaluation(model_path: str, use_mask: bool = False):
    """
    Run a complete evaluation on the validation set.
    Generates:
      - Metrics table
      - ROC curves (logs/roc_curves.png)
      - Confusion matrices per label
    """
    import argparse

    PROC_DIR  = BASE_DIR / "sample_data" / "processed"
    VALID_CSV = PROC_DIR / "processed_valid.csv"

    if not VALID_CSV.exists():
        print("❌ processed_valid.csv not found. Run preprocess_chexpert.py first.")
        sys.exit(1)

    from training.dataset import CheXpertDataset, get_val_transform
    from torch.utils.data import DataLoader
    from training.train import build_model, MaskedBCELoss

    print(f"Loading model from {model_path}...")
    ckpt = torch.load(model_path, map_location="cpu")
    backbone = ckpt.get("backbone", "resnet18")
    model = build_model(backbone=backbone)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_ds = CheXpertDataset(str(VALID_CSV), transform=get_val_transform(), return_mask=use_mask)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    criterion = MaskedBCELoss()

    print("Running evaluation...")
    metrics = evaluate_epoch(model, val_dl, criterion, torch.device("cpu"), use_mask=use_mask)
    print_metrics(metrics)

    # Plot ROC curves
    _plot_roc_curves(model, val_dl, BASE_DIR / "logs" / "roc_curves.png")

    return metrics


def _plot_roc_curves(model, loader, save_path: Path):
    """Plot ROC curves for all 6 labels."""
    try:
        import matplotlib.pyplot as plt

        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                images = batch[0]
                labels = batch[1]
                probs  = torch.sigmoid(model(images))
                all_probs.append(probs.numpy())
                all_labels.append(labels.numpy())

        probs_np  = np.concatenate(all_probs,  axis=0)
        labels_np = np.concatenate(all_labels, axis=0)

        colors = ["#2f81f7","#3fb950","#f85149","#d29922","#bc8cff","#ff7b72"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, (label, color) in enumerate(zip(SELECTED_LABELS, colors)):
            y_true = labels_np[:, i]
            y_prob = probs_np[:, i]
            valid  = ~np.isnan(y_true)
            y_true = y_true[valid]; y_prob = y_prob[valid]

            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc = roc_auc_score(y_true, y_prob)
                axes[i].plot(fpr, tpr, color=color, lw=2, label=f"AUC = {auc:.4f}")
            else:
                axes[i].text(0.5, 0.5, "Insufficient\nclass variety",
                             ha="center", va="center", transform=axes[i].transAxes)

            axes[i].plot([0,1],[0,1],"k--", alpha=0.3)
            axes[i].set_title(f"{label}", fontweight="bold")
            axes[i].set_xlabel("False Positive Rate")
            axes[i].set_ylabel("True Positive Rate")
            axes[i].legend(loc="lower right")
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim([0,1]); axes[i].set_ylim([0,1])

        plt.suptitle("ROC Curves — CheXpert 6-Label Classifier", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(str(save_path), dpi=100, bbox_inches="tight")
        plt.close()
        print(f"[Plot] ROC curves saved to {save_path}")

    except Exception as e:
        print(f"[Plot] Could not generate ROC curves: {e}")


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/chexpert_model.pth")
    parser.add_argument("--ignore", action="store_true")
    args = parser.parse_args()

    full_evaluation(args.model, use_mask=args.ignore)

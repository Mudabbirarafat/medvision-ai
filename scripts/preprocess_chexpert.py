"""
scripts/preprocess_chexpert.py  (FIXED VERSION)
------------------------------------------------
Automatically finds train.csv whether it is at:
  - sample_data/chexpert/train.csv          (old location)
  - sample_data/chexpert/CheXpert-v1.0-small/train.csv   (real CheXpert location)

Run:
  python scripts/preprocess_chexpert.py --max_samples 5000 --uncertainty zeros --no_verify
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ─── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent
CHEXPERT_DIR  = BASE_DIR / "sample_data" / "chexpert"
PROCESSED_DIR = BASE_DIR / "sample_data" / "processed"

SELECTED_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
    "Pneumonia",
]
LABEL_COLS = [f"label_{l.replace(' ', '_')}" for l in SELECTED_LABELS]
IMAGE_SIZE = 224


def find_csv_and_root(chexpert_dir: Path):
    """
    Locate train.csv and valid.csv regardless of where they are.

    CheXpert zip extracts to one of two layouts:
      Layout A (what our zip gave):
          sample_data/chexpert/train.csv
          sample_data/chexpert/CheXpert-v1.0-small/train/...

      Layout B (real Stanford zip):
          sample_data/chexpert/CheXpert-v1.0-small/train.csv
          sample_data/chexpert/CheXpert-v1.0-small/train/...

    Returns (train_csv_path, valid_csv_path, image_root)
    """
    # Try Layout B first (real CheXpert)
    inner = chexpert_dir / "CheXpert-v1.0-small"
    if (inner / "train.csv").exists():
        print(f"  Found CSVs inside CheXpert-v1.0-small/ (real CheXpert layout)")
        return inner / "train.csv", inner / "valid.csv", inner

    # Try Layout A (our generated demo)
    if (chexpert_dir / "train.csv").exists():
        # Check if this CSV has the real number of rows
        df = pd.read_csv(chexpert_dir / "train.csv")
        if len(df) > 1000:
            print(f"  Found CSVs in chexpert/ directly ({len(df):,} rows)")
            return chexpert_dir / "train.csv", chexpert_dir / "valid.csv", chexpert_dir
        else:
            print(f"  WARNING: Found train.csv but only {len(df)} rows — this is the OLD demo CSV.")
            print(f"  Looking for real CheXpert CSV...")

    # Search recursively
    for csv_path in chexpert_dir.rglob("train.csv"):
        df = pd.read_csv(csv_path)
        if len(df) > 1000:
            print(f"  Found real train.csv at: {csv_path} ({len(df):,} rows)")
            root = csv_path.parent
            return csv_path, root / "valid.csv", root

    raise FileNotFoundError(
        f"\n❌ Cannot find real CheXpert train.csv under {chexpert_dir}\n"
        f"   Make sure you extracted CheXpert-v1.0-small.zip into:\n"
        f"   {chexpert_dir}\n"
        f"   Expected structure:\n"
        f"   {chexpert_dir}/CheXpert-v1.0-small/train.csv\n"
        f"   {chexpert_dir}/CheXpert-v1.0-small/train/patient00001/..."
    )


def apply_uncertainty(value: float, strategy: str) -> float:
    if pd.isna(value):
        return 0.0
    if value == -1.0:
        if strategy == "zeros":  return 0.0
        if strategy == "ones":   return 1.0
        if strategy == "ignore": return float("nan")
    return float(value)


def process_split(csv_path, image_root, split_name, strategy,
                  max_samples=None, frontal_only=True, no_verify=False):

    print(f"\n[{split_name.upper()}] Reading {csv_path.name}...")
    df = pd.read_csv(csv_path)
    print(f"   Raw rows         : {len(df):,}")

    # Check required columns
    missing = [l for l in SELECTED_LABELS if l not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Filter frontal only
    if frontal_only and "Frontal/Lateral" in df.columns:
        df = df[df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)
        print(f"   After frontal    : {len(df):,}")

    # Sample
    if max_samples and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"   After sampling   : {len(df):,}")

    records = []
    skipped = 0
    print(f"   Processing with uncertainty='{strategy}'...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split_name}"):
        # Build image path from CSV 'Path' column
        # CheXpert paths look like: CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
        raw_path = str(row["Path"]).replace("\\", "/").lstrip("/")

        # Try multiple path strategies
        candidates = [
            image_root.parent / raw_path,           # relative to chexpert dir parent
            image_root / raw_path,                   # relative to image root
            image_root / Path(*Path(raw_path).parts[1:]),  # strip first folder
            BASE_DIR / raw_path,                     # relative to project root
        ]

        img_path = None
        if not no_verify:
            for c in candidates:
                if c.exists():
                    img_path = c
                    break
            if img_path is None:
                skipped += 1
                continue
        else:
            # Use the most likely path without checking
            img_path = candidates[0]
            if not img_path.exists():
                img_path = candidates[2]

        # Process labels
        rec = {
            "image_path": str(img_path),
            "original_path": str(row["Path"]),
            "split": split_name,
            "sex":  str(row.get("Sex", "Unknown")),
            "age":  int(row.get("Age", 0)) if not pd.isna(row.get("Age", float("nan"))) else 0,
            "view": str(row.get("Frontal/Lateral", "Frontal")),
        }
        for label, col in zip(SELECTED_LABELS, LABEL_COLS):
            rec[col] = apply_uncertainty(row.get(label, float("nan")), strategy)
        records.append(rec)

    result = pd.DataFrame(records)
    print(f"   Valid records    : {len(result):,}")
    if skipped:
        print(f"   Skipped          : {skipped:,} (image not found)")
    return result


def print_summary(train_df, valid_df):
    print(f"\n{'='*60}")
    print(f"  DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Training samples  : {len(train_df):,}")
    print(f"  Validation samples: {len(valid_df):,}")
    print(f"  Total             : {len(train_df)+len(valid_df):,}")
    print(f"  Selected labels   : {len(SELECTED_LABELS)}")
    print()

    all_df = pd.concat([train_df, valid_df])
    total  = len(all_df)
    print(f"  {'Label':<22}  {'Positive':>9}  {'Prevalence':>10}")
    print("  " + "-"*46)
    for label, col in zip(SELECTED_LABELS, LABEL_COLS):
        if col in all_df.columns:
            n_pos = (all_df[col] == 1.0).sum()
            print(f"  {label:<22}  {n_pos:>9,}  {n_pos/total*100:>9.1f}%")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Preprocess CheXpert (fixed version)")
    parser.add_argument("--chexpert_dir",  default=str(CHEXPERT_DIR))
    parser.add_argument("--output_dir",    default=str(PROCESSED_DIR))
    parser.add_argument("--uncertainty",   default="zeros", choices=["zeros","ones","ignore"])
    parser.add_argument("--max_samples",   type=int, default=None,
                        help="Max training samples. Use 5000 for quick demo.")
    parser.add_argument("--max_val",       type=int, default=None)
    parser.add_argument("--frontal_only",  action="store_true", default=True)
    parser.add_argument("--no_verify",     action="store_true", default=False,
                        help="Skip image file existence check (faster)")
    args = parser.parse_args()

    chexpert_dir = Path(args.chexpert_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CheXpert Preprocessing (Fixed Version)")
    print(f"{'='*60}")
    print(f"  Looking for CheXpert in : {chexpert_dir}")

    # Auto-detect CSV location
    train_csv, valid_csv, image_root = find_csv_and_root(chexpert_dir)

    print(f"  Train CSV        : {train_csv}")
    print(f"  Valid CSV        : {valid_csv}")
    print(f"  Image root       : {image_root}")
    print(f"  Uncertainty      : {args.uncertainty}")
    print(f"  Max train samples: {args.max_samples or 'all'}")
    print(f"  Skip img verify  : {args.no_verify}")

    train_df = process_split(train_csv, image_root, "train", args.uncertainty,
                             max_samples=args.max_samples,
                             frontal_only=args.frontal_only,
                             no_verify=args.no_verify)

    valid_df = process_split(valid_csv, image_root, "valid", args.uncertainty,
                             max_samples=args.max_val,
                             frontal_only=args.frontal_only,
                             no_verify=args.no_verify)

    # Save
    train_df.to_csv(output_dir / "processed_train.csv", index=False)
    valid_df.to_csv(output_dir / "processed_valid.csv", index=False)
    pd.concat([train_df, valid_df]).to_csv(output_dir / "processed_dataset.csv", index=False)

    # Label stats
    stats = []
    all_df = pd.concat([train_df, valid_df])
    for label, col in zip(SELECTED_LABELS, LABEL_COLS):
        if col in all_df.columns:
            vals = all_df[col].dropna()
            n_pos = (vals == 1.0).sum()
            n_neg = (vals == 0.0).sum()
            stats.append({"label": label, "column": col,
                          "n_positive": int(n_pos), "n_negative": int(n_neg),
                          "prevalence_pct": round(n_pos/len(all_df)*100, 2)})
    pd.DataFrame(stats).to_csv(output_dir / "label_stats.csv", index=False)

    print_summary(train_df, valid_df)

    print("\n✅ Preprocessing complete!")
    print(f"   {output_dir / 'processed_train.csv'}")
    print(f"   {output_dir / 'processed_valid.csv'}")
    print()
    print("Next step:")
    print("   python training/train.py --epochs 10 --batch_size 16")


if __name__ == "__main__":
    main()

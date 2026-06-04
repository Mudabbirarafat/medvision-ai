# Step 1 — CheXpert Dataset Setup Guide

## What is CheXpert?

CheXpert is a large chest X-ray dataset from Stanford ML Group.
- 224,316 chest radiographs of 65,240 patients
- 14 disease labels per image (multi-label)
- Labels: positive (1), negative (0), uncertain (-1), blank (NaN)
- CheXpert-v1.0-small = downsampled 320×320 version (~11 GB)

## How to Download

### Option A — Stanford Official (Free, requires registration)
1. Go to: https://stanfordmlgroup.github.io/competitions/chexpert/
2. Click "Register" and fill in the form (academic use)
3. You will receive a download link by email (usually within minutes)
4. Download CheXpert-v1.0-small.zip (~11 GB)

### Option B — Kaggle (Faster, needs Kaggle account)
1. Go to: https://www.kaggle.com/datasets/ashery/chexpert
2. Click Download
3. OR use Kaggle CLI:
   ```
   pip install kaggle
   kaggle datasets download -d ashery/chexpert
   ```

### Option C — Use a 200-image Public Sample (for testing only)
If you just want to TEST the pipeline without downloading 11 GB:
```bash
python scripts/download_sample.py
```
This downloads 200 public-domain chest X-rays and creates a compatible CSV.

---

## Exact Folder Structure Required

After downloading, your project must look like this:

```
medvision/
├── sample_data/
│   └── chexpert/
│       ├── CheXpert-v1.0-small/
│       │   ├── train/
│       │   │   ├── patient00001/
│       │   │   │   └── study1/
│       │   │   │       └── view1_frontal.jpg
│       │   │   ├── patient00002/
│       │   │   │   └── study1/
│       │   │   │       └── view1_frontal.jpg
│       │   │   └── ... (64,740 patients)
│       │   └── valid/
│       │       ├── patient00001/
│       │       │   └── study1/
│       │       │       └── view1_frontal.jpg
│       │       └── ... (234 patients)
│       ├── train.csv          ← training labels
│       └── valid.csv          ← validation labels
```

## CSV Format Explanation

### train.csv columns:
| Column | Example | Meaning |
|--------|---------|---------|
| Path | CheXpert-v1.0-small/train/patient00001/... | Relative path to image |
| Sex | Female | Patient sex |
| Age | 68 | Patient age |
| Frontal/Lateral | Frontal | View type |
| AP/PA | PA | Projection |
| No Finding | 0.0 | Label: absent |
| Enlarged Cardiomediastinum | NaN | Label: not mentioned |
| Cardiomegaly | 1.0 | Label: present |
| Lung Opacity | -1.0 | Label: UNCERTAIN |
| ... 14 total label columns | | |

### Label values:
- **1.0**  → Condition IS present (positive)
- **0.0**  → Condition is NOT present (negative)
- **-1.0** → Uncertain (radiologist was not sure)
- **NaN**  → Not mentioned in the report (treated as 0)

---

## Step After Download

Once you have the zip file:
```bash
# Extract to sample_data/chexpert/
unzip CheXpert-v1.0-small.zip -d sample_data/chexpert/

# Verify structure
ls sample_data/chexpert/CheXpert-v1.0-small/
# Should show: train/  valid/  train.csv  valid.csv

# Run preprocessing
python scripts/preprocess_chexpert.py

# Train
python training/train.py

# Evaluate
python training/evaluate.py
```

---

## How many images should I use?

For a CPU laptop, training on ALL 223,415 images takes days.
Use the --max_samples flag to limit:

| Purpose | Recommended samples | Time (CPU) |
|---------|--------------------|-|
| Quick test | 500 | ~10 min |
| Project demo | 5,000 | ~2-3 hours |
| Full project | 20,000–50,000 | ~overnight |
| Full dataset | 223,415 | 2-3 days |

```bash
python training/train.py --max_samples 5000 --epochs 10
```

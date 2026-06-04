"""
tests/test_system.py
---------------------
Comprehensive tests for the real CheXpert-based MedVision system.
Tests every layer: preprocessing, dataset, model, report, QA, API.

Run:
    python -m pytest tests/test_system.py -v
    python -m pytest tests/test_system.py -v -k "test_model"   # filter
"""

import sys, os, io, pytest, numpy as np, torch
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

TRAIN_CSV = BASE / "sample_data" / "processed" / "processed_train.csv"
VALID_CSV  = BASE / "sample_data" / "processed" / "processed_valid.csv"
MODEL_PATH = BASE / "models" / "chexpert_model.pth"
DATA_EXISTS = TRAIN_CSV.exists() and VALID_CSV.exists()
MODEL_EXISTS = MODEL_PATH.exists()

# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dummy_img_pil():
    from PIL import Image
    arr = np.random.randint(30, 200, (320, 320, 3), dtype=np.uint8)
    return Image.fromarray(arr)

@pytest.fixture(scope="module")
def dummy_img_bytes(dummy_img_pil):
    buf = io.BytesIO()
    dummy_img_pil.save(buf, format="JPEG")
    return buf.getvalue()

@pytest.fixture(scope="module")
def model_no_pretrain():
    from training.train import build_model
    from training.dataset import NUM_CLASSES
    m = build_model("resnet18", num_classes=NUM_CLASSES, pretrained=False)
    m.eval()
    return m

@pytest.fixture(scope="module")
def sample_predictions():
    from training.dataset import SELECTED_LABELS
    probs = [0.82, 0.35, 0.61, 0.45, 0.78, 0.29]
    return [
        {"label": l, "probability": p, "confidence_pct": int(p*100),
         "positive": p >= 0.4, "severity": "high" if p >= 0.75 else "moderate" if p >= 0.5 else "low" if p >= 0.4 else "negative"}
        for l, p in zip(SELECTED_LABELS, probs)
    ]

# ─── 1. Dataset & preprocessing ───────────────────────────────────────────────

class TestDataset:
    @pytest.mark.skipif(not DATA_EXISTS, reason="processed CSVs not found")
    def test_csv_loads(self):
        import pandas as pd
        df = pd.read_csv(TRAIN_CSV)
        assert len(df) > 0
        label_cols = [c for c in df.columns if c.startswith("label_")]
        assert len(label_cols) == 6

    @pytest.mark.skipif(not DATA_EXISTS, reason="processed CSVs not found")
    def test_dataset_length(self):
        from training.dataset import CheXpertDataset, get_val_transform
        ds = CheXpertDataset(str(TRAIN_CSV), transform=get_val_transform())
        assert len(ds) > 0

    @pytest.mark.skipif(not DATA_EXISTS, reason="processed CSVs not found")
    def test_dataset_item_shapes(self):
        from training.dataset import CheXpertDataset, get_val_transform, NUM_CLASSES
        ds = CheXpertDataset(str(TRAIN_CSV), transform=get_val_transform())
        img, lbl = ds[0]
        assert img.shape == (3, 224, 224), f"Expected (3,224,224), got {img.shape}"
        assert lbl.shape == (NUM_CLASSES,), f"Expected ({NUM_CLASSES},), got {lbl.shape}"

    @pytest.mark.skipif(not DATA_EXISTS, reason="processed CSVs not found")
    def test_dataset_label_values(self):
        from training.dataset import CheXpertDataset, get_val_transform
        ds = CheXpertDataset(str(TRAIN_CSV), transform=get_val_transform())
        _, lbl = ds[0]
        for v in lbl.tolist():
            assert v in (0.0, 1.0) or (0.0 <= v <= 1.0), f"Unexpected label value: {v}"

    @pytest.mark.skipif(not DATA_EXISTS, reason="processed CSVs not found")
    def test_pos_weights_shape(self):
        from training.dataset import CheXpertDataset, get_val_transform, NUM_CLASSES
        ds = CheXpertDataset(str(TRAIN_CSV), transform=get_val_transform())
        w = ds.get_label_weights()
        assert w.shape == (NUM_CLASSES,)
        assert (w > 0).all(), "All pos_weights should be positive"

    @pytest.mark.skipif(not DATA_EXISTS, reason="processed CSVs not found")
    def test_dataloader_batch(self):
        from training.dataset import get_dataloaders
        tl, vl, pw = get_dataloaders(str(TRAIN_CSV), str(VALID_CSV), batch_size=4, num_workers=0)
        imgs, labels = next(iter(tl))
        assert imgs.shape[0] == 4
        assert imgs.shape[1:] == (3, 224, 224)
        assert labels.shape[1] == 6


# ─── 2. Transforms ────────────────────────────────────────────────────────────

class TestTransforms:
    def test_train_transform(self, dummy_img_pil):
        from training.dataset import get_train_transform
        t = get_train_transform()
        out = t(dummy_img_pil)
        assert out.shape == (3, 224, 224)
        assert out.dtype == torch.float32

    def test_val_transform(self, dummy_img_pil):
        from training.dataset import get_val_transform
        t = get_val_transform()
        out = t(dummy_img_pil)
        assert out.shape == (3, 224, 224)

    def test_val_transform_deterministic(self, dummy_img_pil):
        from training.dataset import get_val_transform
        t = get_val_transform()
        out1 = t(dummy_img_pil)
        out2 = t(dummy_img_pil)
        assert torch.allclose(out1, out2), "Val transform must be deterministic"

    def test_normalization_range(self, dummy_img_pil):
        """After ImageNet normalization values should extend beyond [0,1]."""
        from training.dataset import get_val_transform
        t = get_val_transform()
        out = t(dummy_img_pil)
        assert out.min() < 0 or out.max() > 1, "Normalization should shift values from [0,1]"

    def test_inference_transform(self, dummy_img_pil):
        from training.dataset import get_inference_transform
        t = get_inference_transform()
        out = t(dummy_img_pil)
        assert out.shape == (3, 224, 224)


# ─── 3. Model architecture ────────────────────────────────────────────────────

class TestModel:
    def test_resnet18_output_shape(self, model_no_pretrain):
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model_no_pretrain(x)
        assert out.shape == (2, 6), f"Expected (2,6), got {out.shape}"

    def test_mobilenet_output_shape(self):
        from training.train import build_model
        m = build_model("mobilenet_v2", pretrained=False)
        m.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (1, 6)

    def test_sigmoid_range(self, model_no_pretrain):
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            logits = model_no_pretrain(x)
            probs  = torch.sigmoid(logits)
        assert probs.min() >= 0.0 and probs.max() <= 1.0

    def test_batch_size_1(self, model_no_pretrain):
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model_no_pretrain(x)
        assert out.shape == (1, 6)

    def test_batch_size_16(self, model_no_pretrain):
        x = torch.randn(16, 3, 224, 224)
        with torch.no_grad():
            out = model_no_pretrain(x)
        assert out.shape == (16, 6)

    def test_eval_deterministic(self, model_no_pretrain):
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            o1 = model_no_pretrain(x)
            o2 = model_no_pretrain(x)
        assert torch.allclose(o1, o2), "Model in eval mode must be deterministic"

    def test_no_gradient_leak(self, model_no_pretrain):
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = torch.sigmoid(model_no_pretrain(x))
        assert not out.requires_grad

    def test_total_params(self, model_no_pretrain):
        total = sum(p.numel() for p in model_no_pretrain.parameters())
        assert total > 10_000_000, "ResNet18 should have >10M params"
        assert total < 15_000_000, "ResNet18 should have <15M params"


# ─── 4. Loss function ─────────────────────────────────────────────────────────

class TestLoss:
    def test_masked_bce_scalar(self):
        from training.train import MaskedBCELoss
        crit = MaskedBCELoss()
        logits = torch.randn(4, 6)
        targets = torch.randint(0, 2, (4, 6)).float()
        loss = crit(logits, targets)
        assert loss.ndim == 0 and loss.item() > 0

    def test_masked_bce_with_pos_weight(self):
        from training.train import MaskedBCELoss
        pw = torch.ones(6) * 3.0
        crit = MaskedBCELoss(pos_weight=pw)
        logits = torch.zeros(4, 6)
        targets = torch.ones(4, 6)
        loss = crit(logits, targets)
        assert loss.item() > 0

    def test_masked_bce_with_mask(self):
        from training.train import MaskedBCELoss
        crit = MaskedBCELoss()
        logits = torch.randn(4, 6)
        targets = torch.randint(0, 2, (4, 6)).float()
        mask = torch.ones(4, 6, dtype=torch.bool)
        mask[:, 2] = False  # Ignore label index 2
        loss_masked   = crit(logits, targets, mask)
        loss_unmasked = crit(logits, targets)
        assert loss_masked.item() != loss_unmasked.item()


# ─── 5. Training step ─────────────────────────────────────────────────────────

class TestTrainingStep:
    def test_single_gradient_step(self):
        from training.train import build_model, MaskedBCELoss
        from training.dataset import NUM_CLASSES
        import torch.optim as optim

        model = build_model("resnet18", num_classes=NUM_CLASSES, pretrained=False)
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        criterion = MaskedBCELoss()

        imgs   = torch.randn(4, 3, 224, 224)
        labels = torch.randint(0, 2, (4, NUM_CLASSES)).float()

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()

        # Check gradients flow
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert len(grad_norms) > 0, "No gradients computed"
        assert any(g > 0 for g in grad_norms), "All gradients are zero"

        optimizer.step()  # Should not raise


# ─── 6. Checkpoint save / load ────────────────────────────────────────────────

class TestCheckpoint:
    def test_save_and_load(self, tmp_path, model_no_pretrain):
        from training.train import save_checkpoint, build_model
        from training.dataset import NUM_CLASSES, SELECTED_LABELS
        import torch.optim as optim

        opt = optim.AdamW(model_no_pretrain.parameters(), lr=1e-3)
        path = tmp_path / "test_ckpt.pth"
        save_checkpoint(model_no_pretrain, opt, 5, {"mean_auc": 0.75},
                        path, "resnet18", SELECTED_LABELS)
        assert path.exists()

        ckpt = torch.load(str(path), map_location="cpu")
        assert ckpt["epoch"] == 5
        assert ckpt["backbone"] == "resnet18"
        assert len(ckpt["selected_labels"]) == 6

        # Reload
        m2 = build_model("resnet18", num_classes=NUM_CLASSES, pretrained=False)
        m2.load_state_dict(ckpt["model_state_dict"])
        m2.eval()

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            o1 = model_no_pretrain(x)
            o2 = m2(x)
        assert torch.allclose(o1, o2), "Loaded model outputs differ from original"


# ─── 7. Evaluation metrics ────────────────────────────────────────────────────

class TestEvaluation:
    @pytest.mark.skipif(not DATA_EXISTS, reason="processed CSVs not found")
    def test_evaluate_epoch_keys(self, model_no_pretrain):
        from training.dataset import CheXpertDataset, get_val_transform
        from training.train import MaskedBCELoss
        from training.evaluate import evaluate_epoch
        from torch.utils.data import DataLoader

        ds = CheXpertDataset(str(VALID_CSV), transform=get_val_transform())
        dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        crit = MaskedBCELoss()
        metrics = evaluate_epoch(model_no_pretrain, dl, crit, torch.device("cpu"))

        for key in ["loss", "mean_auc", "mean_f1", "mean_acc",
                    "per_class_auc", "per_class_f1", "n_samples"]:
            assert key in metrics, f"Missing key: {key}"

    @pytest.mark.skipif(not DATA_EXISTS, reason="processed CSVs not found")
    def test_evaluate_metrics_range(self, model_no_pretrain):
        from training.dataset import CheXpertDataset, get_val_transform
        from training.train import MaskedBCELoss
        from training.evaluate import evaluate_epoch
        from torch.utils.data import DataLoader

        ds = CheXpertDataset(str(VALID_CSV), transform=get_val_transform())
        dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        crit = MaskedBCELoss()
        metrics = evaluate_epoch(model_no_pretrain, dl, crit, torch.device("cpu"))

        assert metrics["loss"] > 0
        if not np.isnan(metrics["mean_auc"]):
            assert 0.0 <= metrics["mean_auc"] <= 1.0
        assert 0.0 <= metrics["mean_f1"] <= 1.0


# ─── 8. Report generator ──────────────────────────────────────────────────────

class TestReportGenerator:
    def test_generate_with_findings(self, sample_predictions):
        from model_service.report_generator import generate_report
        pos = [p["label"] for p in sample_predictions if p["positive"]]
        report, text = generate_report(sample_predictions, pos)
        for key in ["indication", "technique", "findings", "impression",
                    "recommendations", "disclaimer"]:
            assert key in report, f"Missing report key: {key}"
        assert isinstance(text, str) and len(text) > 100

    def test_generate_no_findings(self):
        from model_service.report_generator import generate_report
        from training.dataset import SELECTED_LABELS
        preds = [{"label": l, "probability": 0.1, "confidence_pct": 10,
                  "positive": False, "severity": "negative"}
                 for l in SELECTED_LABELS]
        report, text = generate_report(preds, [])
        assert "normal" in report["findings"].lower() or "no" in report["findings"].lower()

    def test_report_text_has_sections(self, sample_predictions):
        from model_service.report_generator import generate_report
        pos = [p["label"] for p in sample_predictions if p["positive"]]
        _, text = generate_report(sample_predictions, pos)
        for section in ["TECHNIQUE", "FINDINGS", "IMPRESSION", "RECOMMENDATIONS"]:
            assert section in text, f"Missing section: {section}"

    def test_report_disclaimer(self, sample_predictions):
        from model_service.report_generator import generate_report
        pos = [p["label"] for p in sample_predictions if p["positive"]]
        report, _ = generate_report(sample_predictions, pos)
        assert "DISCLAIMER" in report["disclaimer"].upper() or "research" in report["disclaimer"].lower()


# ─── 9. QA engine ─────────────────────────────────────────────────────────────

class TestQAEngine:
    def test_pneumonia_positive(self, sample_predictions):
        from model_service.qa_engine import answer_question
        # Pneumonia has 0.29 (negative) in sample_predictions
        # Let's force a positive case
        preds = [dict(p) for p in sample_predictions]
        for p in preds:
            if p["label"] == "Pneumonia":
                p["probability"] = 0.82
                p["confidence_pct"] = 82
                p["positive"] = True
                p["severity"] = "high"
        result = answer_question("Is there pneumonia?", preds)
        assert "YES" in result["answer"]
        assert "Pneumonia" in result.get("relevant_findings", [])

    def test_pneumonia_negative(self, sample_predictions):
        from model_service.qa_engine import answer_question
        preds = [dict(p) for p in sample_predictions]
        for p in preds:
            if p["label"] == "Pneumonia":
                p["probability"] = 0.15
                p["confidence_pct"] = 15
                p["positive"] = False
        result = answer_question("Is there pneumonia?", preds)
        assert "NO" in result["answer"]

    def test_normal_question_with_findings(self, sample_predictions):
        from model_service.qa_engine import answer_question
        result = answer_question("Is this X-ray normal?", sample_predictions)
        assert "NOT" in result["answer"] or "not normal" in result["answer"].lower()

    def test_list_question(self, sample_predictions):
        from model_service.qa_engine import answer_question
        result = answer_question("What are all the findings?", sample_predictions)
        assert isinstance(result["answer"], str) and len(result["answer"]) > 20

    def test_primary_diagnosis(self, sample_predictions):
        from model_service.qa_engine import answer_question
        result = answer_question("What is the most likely diagnosis?", sample_predictions)
        assert isinstance(result["answer"], str) and len(result["answer"]) > 10

    def test_response_keys(self, sample_predictions):
        from model_service.qa_engine import answer_question
        result = answer_question("Show me all findings", sample_predictions)
        for k in ["question", "answer", "confidence", "relevant_findings", "disclaimer"]:
            assert k in result, f"Missing QA key: {k}"

    def test_confidence_range(self, sample_predictions):
        from model_service.qa_engine import answer_question
        result = answer_question("Is there cardiomegaly?", sample_predictions)
        assert 0.0 <= result["confidence"] <= 1.0


# ─── 10. Preprocessing script ─────────────────────────────────────────────────

class TestPreprocessScript:
    def test_uncertainty_zeros(self):
        from scripts.preprocess_chexpert import apply_uncertainty_strategy
        assert apply_uncertainty_strategy(-1.0, "zeros") == 0.0
        assert apply_uncertainty_strategy(1.0,  "zeros") == 1.0
        assert apply_uncertainty_strategy(0.0,  "zeros") == 0.0
        assert apply_uncertainty_strategy(float("nan"), "zeros") == 0.0

    def test_uncertainty_ones(self):
        from scripts.preprocess_chexpert import apply_uncertainty_strategy
        assert apply_uncertainty_strategy(-1.0, "ones") == 1.0
        assert apply_uncertainty_strategy(1.0,  "ones") == 1.0

    def test_uncertainty_ignore(self):
        from scripts.preprocess_chexpert import apply_uncertainty_strategy
        import math
        result = apply_uncertainty_strategy(-1.0, "ignore")
        assert math.isnan(result), "ignore strategy should return NaN"

    def test_unknown_strategy(self):
        from scripts.preprocess_chexpert import apply_uncertainty_strategy
        with pytest.raises(ValueError):
            apply_uncertainty_strategy(-1.0, "magic")


# ─── 11. Full pipeline integration test ───────────────────────────────────────

class TestIntegration:
    @pytest.mark.skipif(not DATA_EXISTS, reason="processed CSVs not found")
    def test_full_inference_pipeline(self, dummy_img_bytes):
        """Image bytes → preprocess → model → postprocess → report → QA."""
        from training.dataset import get_inference_transform, SELECTED_LABELS
        from training.train import build_model, MaskedBCELoss
        from model_service.report_generator import generate_report
        from model_service.qa_engine import answer_question
        from PIL import Image

        # 1. Preprocess
        transform = get_inference_transform()
        img = Image.open(io.BytesIO(dummy_img_bytes)).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        assert tensor.shape == (1, 3, 224, 224)

        # 2. Forward pass
        model = build_model("resnet18", num_classes=6, pretrained=False)
        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.sigmoid(logits)[0]
        assert probs.shape == (6,)

        # 3. Build predictions
        THRESHOLD = 0.4
        preds = [
            {"label": l, "probability": round(p.item(), 4),
             "confidence_pct": int(p.item() * 100),
             "positive": p.item() >= THRESHOLD,
             "severity": "high" if p.item() >= 0.75 else "moderate" if p.item() >= 0.5 else "low" if p.item() >= THRESHOLD else "negative"}
            for l, p in zip(SELECTED_LABELS, probs)
        ]
        pos = [p["label"] for p in preds if p["positive"]]

        # 4. Report
        report, text = generate_report(preds, pos)
        assert "findings" in report
        assert isinstance(text, str)

        # 5. QA
        qa = answer_question("Is there pneumonia?", preds)
        assert "answer" in qa

        print("\n[Integration] Full pipeline test PASSED ✓")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

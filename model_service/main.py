"""
model_service/main.py
----------------------
Step 6: Updated FastAPI model service for real CheXpert-trained model.

Endpoints:
  GET  /health   → Check if model is loaded and which labels it serves
  POST /predict  → Upload image → disease probabilities + confidence scores
  POST /report   → Upload image → structured diagnostic report
  POST /qa       → Upload image + question → clinical QA answer

CPU inference time: ~1-4 seconds per image (ResNet18, no GPU)

Run:
    cd model_service
    uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"""

import os
import io
import sys
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# ─── Path setup ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from training.dataset import (
    SELECTED_LABELS, LABEL_COLS, NUM_CLASSES,
    get_inference_transform,
)
from model_service.report_generator import generate_report
from model_service.qa_engine import answer_question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH  = BASE_DIR / "models" / "chexpert_model.pth"
THRESHOLD   = float(os.getenv("INFERENCE_THRESHOLD", "0.4"))
DEVICE      = torch.device("cpu")

# ─── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MedVision Model Service",
    description="CheXpert multi-label chest X-ray classifier — REST API",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global model state ────────────────────────────────────────────────────────
MODEL: Optional[nn.Module] = None
BACKBONE: str = "resnet18"
_transform = get_inference_transform()


def _load_model() -> nn.Module:
    """
    Load the trained CheXpert model.
    Falls back to an untrained model (random weights) if checkpoint not found.
    This allows the service to start even before training is complete.
    """
    from training.train import build_model

    if MODEL_PATH.exists():
        try:
            ckpt = torch.load(str(MODEL_PATH), map_location="cpu")
            backbone = ckpt.get("backbone", "resnet18")
            model = build_model(backbone=backbone, num_classes=NUM_CLASSES)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            logger.info(
                f"✓ Loaded model: {MODEL_PATH.name} | "
                f"backbone={backbone} | "
                f"labels={ckpt.get('selected_labels', SELECTED_LABELS)}"
            )
            global BACKBONE
            BACKBONE = backbone
            return model
        except Exception as e:
            logger.warning(f"Could not load model weights: {e}")
            logger.warning("Starting with random weights (demo mode). Run training first.")
    else:
        logger.warning(
            f"No model found at {MODEL_PATH}. "
            f"Run: python training/train.py --max_samples 500 --epochs 3"
        )

    # Demo mode: random weights
    from training.train import build_model
    model = build_model(backbone="resnet18", num_classes=NUM_CLASSES)
    model.eval()
    return model


@app.on_event("startup")
async def startup():
    global MODEL
    logger.info("Loading MedVision model service...")
    MODEL = _load_model()
    logger.info(f"Model service ready. Labels: {SELECTED_LABELS}")


# ─── Schemas ───────────────────────────────────────────────────────────────────

class PredictionItem(BaseModel):
    label: str
    probability: float
    confidence_pct: int
    positive: bool
    severity: str        # "high" | "moderate" | "low" | "negative"


class PredictResponse(BaseModel):
    status: str
    model_version: str
    backbone: str
    inference_time_ms: float
    threshold: float
    labels_used: List[str]
    predictions: List[PredictionItem]
    positive_findings: List[str]
    any_finding: bool


class ReportResponse(BaseModel):
    status: str
    positive_findings: List[str]
    report: Dict
    report_text: str


class QAResponse(BaseModel):
    status: str
    question: str
    answer: str
    confidence: float
    relevant_findings: List[str]
    disclaimer: str


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _preprocess_image(raw_bytes: bytes) -> torch.Tensor:
    """Convert raw image bytes to a (1, 3, 224, 224) float tensor."""
    try:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot read image: {e}")
    tensor = _transform(img)       # (3, 224, 224)
    return tensor.unsqueeze(0)     # (1, 3, 224, 224)


def _severity(prob: float) -> str:
    if prob >= 0.75:  return "high"
    if prob >= 0.50:  return "moderate"
    if prob >= 0.40:  return "low"
    return "negative"


def _run_inference(img_tensor: torch.Tensor) -> List[Dict]:
    """Run forward pass and return sorted predictions list."""
    with torch.no_grad():
        logits = MODEL(img_tensor)            # (1, 6)
        probs  = torch.sigmoid(logits)[0]     # (6,)

    predictions = []
    for label, prob in zip(SELECTED_LABELS, probs.tolist()):
        predictions.append({
            "label":          label,
            "probability":    round(prob, 4),
            "confidence_pct": int(prob * 100),
            "positive":       prob >= THRESHOLD,
            "severity":       _severity(prob),
        })

    # Sort by probability descending
    predictions.sort(key=lambda x: x["probability"], reverse=True)
    return predictions


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "model_loaded":  MODEL is not None,
        "model_path":    str(MODEL_PATH),
        "model_exists":  MODEL_PATH.exists(),
        "backbone":      BACKBONE,
        "labels":        SELECTED_LABELS,
        "num_classes":   NUM_CLASSES,
        "threshold":     THRESHOLD,
        "device":        "cpu",
        "version":       "2.0.0",
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(..., description="Chest X-ray image (JPEG/PNG)"),
):
    """
    Classify chest conditions from an uploaded medical image.

    Returns probabilities for 6 conditions:
      Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion, Pneumonia

    Confidence interpretation:
      ≥75% → High confidence positive
      50–74% → Moderate confidence positive
      40–49% → Low confidence positive (above threshold)
      <40% → Negative (not detected)
    """
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")

    raw = await image.read()
    if not raw:
        raise HTTPException(400, "Empty file")

    t0 = time.perf_counter()
    img_tensor  = _preprocess_image(raw)
    predictions = _run_inference(img_tensor)
    elapsed_ms  = (time.perf_counter() - t0) * 1000

    pos_findings = [p["label"] for p in predictions if p["positive"]]

    logger.info(
        f"predict | {image.filename} | "
        f"positive={pos_findings} | {elapsed_ms:.0f}ms"
    )

    return {
        "status":           "success",
        "model_version":    f"MedVision-CheXpert-{BACKBONE}-v2.0",
        "backbone":         BACKBONE,
        "inference_time_ms": round(elapsed_ms, 1),
        "threshold":        THRESHOLD,
        "labels_used":      SELECTED_LABELS,
        "predictions":      predictions,
        "positive_findings": pos_findings,
        "any_finding":      len(pos_findings) > 0,
    }


@app.post("/report", response_model=ReportResponse)
async def report(
    image: UploadFile = File(...),
    clinical_context: Optional[str] = Form(None),
    patient_id: Optional[str] = Form(None),
):
    """Generate a structured natural-language diagnostic report."""
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")

    raw = await image.read()
    img_tensor  = _preprocess_image(raw)
    predictions = _run_inference(img_tensor)
    pos_findings = [p["label"] for p in predictions if p["positive"]]

    report_dict, report_text = generate_report(
        predictions=predictions,
        positive_findings=pos_findings,
        patient_id=patient_id,
        clinical_context=clinical_context,
    )

    return {
        "status":           "success",
        "positive_findings": pos_findings,
        "report":           report_dict,
        "report_text":      report_text,
    }


@app.post("/qa", response_model=QAResponse)
async def qa(
    image: UploadFile = File(...),
    question: str = Form(...),
):
    """Answer a clinical question about the uploaded image."""
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")
    if not question.strip():
        raise HTTPException(400, "Question cannot be empty")

    raw = await image.read()
    img_tensor  = _preprocess_image(raw)
    predictions = _run_inference(img_tensor)

    result = answer_question(question, predictions)
    logger.info(f"qa | Q: {question[:50]} | A: {result['answer'][:60]}")

    return {"status": "success", **result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)

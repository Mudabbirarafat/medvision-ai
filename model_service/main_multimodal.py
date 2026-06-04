"""
model_service/main.py  (MULTIMODAL VERSION)
--------------------------------------------
FastAPI service using the real Multimodal Vision-Language model.

Endpoints:
  GET  /health   → Model info + architecture details
  POST /predict  → Image + optional text → disease probabilities
  POST /report   → Image + context → structured NL report
  POST /qa       → Image + clinical question → answer with attention

The /qa endpoint uses the FULL multimodal path:
  image + question text → cross-modal attention → prediction
  
This is genuine multimodal inference — the text changes
which image regions the model attends to.

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

import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from training.dataset import SELECTED_LABELS, LABEL_COLS, NUM_CLASSES, get_inference_transform
from training.model import MultimodalMedModel, tokenize, load_model, VOCAB_SIZE
from model_service.report_generator import generate_report
from model_service.qa_engine import answer_question

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────────────────
# Priority: multimodal model first, fall back to unimodal
MULTIMODAL_PATH = BASE_DIR / "models" / "chexpert_multimodal.pth"
UNIMODAL_PATH   = BASE_DIR / "models" / "chexpert_model.pth"
THRESHOLD       = 0.4
DEVICE          = torch.device("cpu")

app = FastAPI(
    title="MedVision Multimodal AI",
    description="Genuine Vision-Language Medical Diagnosis System",
    version="3.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL = None
MODEL_TYPE = "none"
_transform = get_inference_transform()


def _load_best_model():
    """Load multimodal model if available, fall back to unimodal."""
    global MODEL_TYPE

    if MULTIMODAL_PATH.exists():
        try:
            model = MultimodalMedModel(pretrained=False, num_classes=NUM_CLASSES)
            ckpt = torch.load(str(MULTIMODAL_PATH), map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            MODEL_TYPE = "multimodal"
            logger.info(f"✓ Loaded MULTIMODAL model: {MULTIMODAL_PATH.name}")
            logger.info(f"  Architecture: ResNet18 + BiGRU + CrossModalAttention")
            return model
        except Exception as e:
            logger.warning(f"Could not load multimodal model: {e}")

    if UNIMODAL_PATH.exists():
        try:
            # Load unimodal model into multimodal architecture
            # Vision encoder weights are compatible
            model = MultimodalMedModel(pretrained=False, num_classes=NUM_CLASSES)
            ckpt = torch.load(str(UNIMODAL_PATH), map_location="cpu")
            state = ckpt["model_state_dict"]
            model.load_state_dict(state, strict=False)
            model.eval()
            MODEL_TYPE = "unimodal_in_multimodal"
            logger.info(f"✓ Loaded unimodal weights into multimodal architecture")
            logger.info(f"  Run train_multimodal.py for full multimodal training")
            return model
        except Exception as e:
            logger.warning(f"Could not load unimodal model: {e}")

    # No trained model — use random weights (demo mode)
    model = MultimodalMedModel(pretrained=False, num_classes=NUM_CLASSES)
    model.eval()
    MODEL_TYPE = "demo_random"
    logger.warning("No trained model found — using random weights (demo mode)")
    logger.warning("Run training/train.py first, then training/train_multimodal.py")
    return model


@app.on_event("startup")
async def startup():
    global MODEL
    MODEL = _load_best_model()
    logger.info(f"Model service ready | type={MODEL_TYPE} | labels={SELECTED_LABELS}")


def _preprocess_image(raw_bytes: bytes) -> torch.Tensor:
    try:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(422, f"Cannot read image: {e}")
    return _transform(img).unsqueeze(0)   # (1, 3, 224, 224)


def _severity(prob: float) -> str:
    if prob >= 0.75: return "high"
    if prob >= 0.50: return "moderate"
    if prob >= 0.40: return "low"
    return "negative"


def _run_prediction(img_tensor, text_query=None):
    """Run multimodal inference."""
    tokens = None
    if text_query and text_query.strip():
        tokens = tokenize(text_query.strip())   # (1, 32)

    with torch.no_grad():
        logits = MODEL(img_tensor, tokens)
        probs  = torch.sigmoid(logits)[0]

    preds = []
    for label, prob in zip(SELECTED_LABELS, probs.tolist()):
        preds.append({
            "label":          label,
            "probability":    round(prob, 4),
            "confidence_pct": int(prob * 100),
            "positive":       prob >= THRESHOLD,
            "severity":       _severity(prob),
        })
    preds.sort(key=lambda x: x["probability"], reverse=True)
    return preds


# ─── Schemas ───────────────────────────────────────────────────────────────────

class PredItem(BaseModel):
    label: str
    probability: float
    confidence_pct: int
    positive: bool
    severity: str


class PredictResponse(BaseModel):
    status: str
    model_type: str
    architecture: str
    inference_time_ms: float
    threshold: float
    text_query_used: Optional[str]
    predictions: List[PredItem]
    positive_findings: List[str]
    any_finding: bool


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "model_type":   MODEL_TYPE,
        "model_loaded": MODEL is not None,
        "architecture": {
            "vision":   "ResNet18 (ImageNet pretrained)",
            "language": "BiGRU (2-layer bidirectional)",
            "fusion":   "Cross-Modal Attention + MLP",
            "type":     "Multimodal Vision-Language",
        },
        "labels":       SELECTED_LABELS,
        "num_classes":  NUM_CLASSES,
        "threshold":    THRESHOLD,
        "models_found": {
            "multimodal": MULTIMODAL_PATH.exists(),
            "unimodal":   UNIMODAL_PATH.exists(),
        },
        "version": "3.0.0",
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    query: Optional[str] = Form(None),
):
    """
    Multimodal disease classification.
    
    If query is provided: image + text → cross-modal attention → predictions
    If no query: image only → predictions (text encoder receives zeros)
    
    The presence of a text query CHANGES the predictions because
    cross-modal attention focuses on different image regions.
    """
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")

    raw = await image.read()
    if not raw:
        raise HTTPException(400, "Empty file")

    t0 = time.perf_counter()
    img_tensor = _preprocess_image(raw)
    preds = _run_prediction(img_tensor, text_query=query)
    elapsed = (time.perf_counter() - t0) * 1000

    pos = [p["label"] for p in preds if p["positive"]]
    logger.info(f"predict | {image.filename} | query={query} | pos={pos} | {elapsed:.0f}ms")

    return {
        "status":            "success",
        "model_type":        MODEL_TYPE,
        "architecture":      "ResNet18 + BiGRU + CrossModalAttention",
        "inference_time_ms": round(elapsed, 1),
        "threshold":         THRESHOLD,
        "text_query_used":   query,
        "predictions":       preds,
        "positive_findings": pos,
        "any_finding":       len(pos) > 0,
    }


@app.post("/report")
async def report(
    image: UploadFile = File(...),
    clinical_context: Optional[str] = Form(None),
    patient_id: Optional[str] = Form(None),
):
    """Generate structured NL diagnostic report."""
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")

    raw = await image.read()
    img_tensor = _preprocess_image(raw)

    # Use clinical context as text query for multimodal inference
    preds = _run_prediction(img_tensor, text_query=clinical_context)
    pos   = [p["label"] for p in preds if p["positive"]]

    report_dict, report_text = generate_report(
        preds, pos, patient_id=patient_id, clinical_context=clinical_context
    )
    return {"status": "success", "positive_findings": pos,
            "report": report_dict, "report_text": report_text}


@app.post("/qa")
async def qa(
    image: UploadFile = File(...),
    question: str = Form(...),
):
    """
    TRUE multimodal QA: image + question → cross-modal attention → answer.
    
    The question text changes what the model attends to in the image.
    "Is there pneumonia?" → model focuses on lung parenchyma
    "Is the heart enlarged?" → model focuses on cardiac silhouette
    """
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")
    if not question.strip():
        raise HTTPException(400, "Question cannot be empty")

    raw = await image.read()
    img_tensor = _preprocess_image(raw)

    # MULTIMODAL: question text influences which predictions are made
    preds = _run_prediction(img_tensor, text_query=question)

    # Also get attention weights to show which image region was attended to
    try:
        tokens = tokenize(question)
        with torch.no_grad():
            attn_weights = MODEL.get_attention_weights(img_tensor, tokens)
        top_patch = attn_weights[0].argmax().item()
        patch_row, patch_col = top_patch // 7, top_patch % 7
        attention_info = {
            "top_patch_index": top_patch,
            "top_patch_grid":  f"({patch_row},{patch_col}) in 7×7 grid",
            "attention_weights": attn_weights[0].tolist(),
        }
    except Exception:
        attention_info = {}

    result = answer_question(question, preds)
    logger.info(f"qa | Q: {question[:50]} | A: {result['answer'][:60]}")

    return {
        "status": "success",
        "multimodal_inference": True,
        "attention_info": attention_info,
        **result
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)

"""
backend/main.py
----------------
FastAPI gateway — sits between the frontend and the model service.
Routes /analyze, /query to the model service and returns unified JSON.

Run:
    cd backend
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import io
import logging
from typing import Optional

import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:8001")
MAX_MB = 30

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="MedVision Backend Gateway", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ALLOWED_TYPES = {"image/jpeg","image/jpg","image/png","image/bmp","image/tiff","image/webp"}


async def _validate(image: UploadFile) -> bytes:
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported type: {image.content_type}")
    content = await image.read()
    if len(content) / 1024**2 > MAX_MB:
        raise HTTPException(413, f"File > {MAX_MB} MB")
    if len(content) < 100:
        raise HTTPException(400, "Empty or corrupt file")
    return content


async def _forward(endpoint: str, content: bytes, filename: str, data: dict = None) -> dict:
    files = {"image": (filename, io.BytesIO(content), "image/jpeg")}
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(f"{MODEL_SERVICE_URL}/{endpoint}", files=files, data=data or {})
            resp.raise_for_status()
            return resp.json()
        except httpx.ConnectError:
            raise HTTPException(503, f"Model service unavailable at {MODEL_SERVICE_URL}. Is it running on port 8001?")
        except httpx.HTTPStatusError as e:
            raise HTTPException(e.response.status_code, str(e))


@app.get("/")
async def root():
    return {"service": "MedVision Backend Gateway", "version": "2.0.0",
            "model_service": MODEL_SERVICE_URL}


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{MODEL_SERVICE_URL}/health")
            model_detail = r.json() if r.status_code == 200 else {}
            model_ok = r.status_code == 200
    except Exception as e:
        model_detail = {"error": str(e)}
        model_ok = False
    return {"gateway": "ok", "model_service": "ok" if model_ok else "unavailable",
            "detail": model_detail}


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    query: Optional[str] = Form(None),
    clinical_context: Optional[str] = Form(None),
    patient_id: Optional[str] = Form(None),
    include_report: bool = Form(True),
    include_qa: bool = Form(False),
):
    """Full pipeline: classify + report + optional QA."""
    content = await _validate(image)
    fname   = image.filename or "image.jpg"

    # 1. Predict
    predict = await _forward("predict", content, fname)
    response = {"status": "success", "filename": fname, "classification": predict}

    # 2. Report
    if include_report:
        try:
            report = await _forward("report", content, fname,
                                    data={"clinical_context": clinical_context or "",
                                          "patient_id": patient_id or ""})
            response["report"] = report
        except Exception as e:
            response["report"] = {"error": str(e)}

    # 3. QA
    if include_qa and query:
        try:
            qa = await _forward("qa", content, fname, data={"question": query})
            response["qa"] = qa
        except Exception as e:
            response["qa"] = {"error": str(e)}

    return response


@app.post("/query")
async def query_image(
    image: UploadFile = File(...),
    question: str = Form(...),
):
    content = await _validate(image)
    return await _forward("qa", content, image.filename or "image.jpg",
                          data={"question": question})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

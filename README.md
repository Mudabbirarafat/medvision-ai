# 🏥 MedVision AI

[![CI](https://github.com/Mudabbirarafat/medvision-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Mudabbirarafat/medvision-ai/actions/workflows/ci.yml)

An AI-powered Medical Imaging and Diagnostic Assistant that leverages Deep Learning, Computer Vision, and Multimodal AI techniques to assist in chest X-ray analysis, report generation, and clinical decision support.

## 🚀 Overview

MedVision AI is designed to help healthcare professionals analyze chest X-ray images using state-of-the-art deep learning models. The system combines medical image processing, automated report generation, and intelligent question-answering capabilities to provide a comprehensive diagnostic workflow.

The project is built using Python, PyTorch, FastAPI, and modern AI/ML techniques.

---

## ✨ Features

### 🩻 Medical Image Analysis

* Chest X-ray disease detection
* Deep learning-based image classification
* Automated abnormality identification
* Support for CheXpert dataset

### 📄 AI Report Generation

* Automatic radiology-style report generation
* Clinical summary creation
* Diagnostic findings extraction

### 🤖 Medical Question Answering

* Context-aware medical Q&A system
* Interactive diagnostic assistance
* Knowledge-assisted response generation

### 🌐 Full-Stack Architecture

* Frontend interface for image upload and analysis
* Backend API services
* Dedicated model-serving layer
* Modular and scalable architecture

---

## 🏗️ Project Architecture

Frontend → Backend API → Model Service → Deep Learning Models

Components:

* Frontend (User Interface)
* Backend (API Gateway)
* Model Service (Inference Engine)
* Report Generator
* QA Engine
* Training Pipeline

---

## 🛠️ Tech Stack

### AI / Machine Learning

* PyTorch
* Deep Learning
* Computer Vision
* Multimodal AI

### Backend

* Python
* FastAPI

### Frontend

* HTML
* CSS
* JavaScript

### Data Processing

* Pandas
* NumPy

### Testing

* PyTest

---

## 📂 Project Structure

```text
medvision-ai/
│
├── backend/
├── frontend/
├── model_service/
├── training/
├── scripts/
├── tests/
├── utils/
├── requirements.txt
└── README.md
```

---
## Model Architecture

Input Image (chest X-ray JPEG/PNG)
        │
  [scripts/preprocess_chexpert.py]
  → Resize + normalize → processed CSV
        │
  [training/dataset.py]
  → CheXpertDataset: load JPEG → tensor
  → ImageNet normalization (mean/std)
  → Train: augment (flip, rotate, jitter)
        │
  [training/train.py]
  ┌──────────────────────────────────────┐
  │  ResNet18 (ImageNet pretrained)      │
  │  → Remove FC layer                  │
  │  → feature vector (512-dim)         │
  │  → Linear(512 → 256) + ReLU + Drop  │
  │  → Linear(256 → 6) [no sigmoid]     │
  └──────────────────────────────────────┘
        │  logits (6,)
  BCEWithLogitsLoss(pos_weight=...)
  AdamW + CosineAnnealingLR
        │
  models/chexpert_model.pth
        │
  [model_service/main.py]  FastAPI :8001
  → sigmoid(logits) → probabilities
  → threshold (0.4) → positive/negative
  → report_generator.py → NL report
  → qa_engine.py → Q&A
        │
  [backend/main.py]  FastAPI :8000
        │
  [frontend/index.html]  Browser

## 📊 Dataset

This project uses the CheXpert dataset for chest X-ray analysis and model training.

Dataset and trained model weights are not included in this repository due to size limitations.

---

## ⚙️ Installation

```bash
git clone https://github.com/Mudabbirarafat/medvision-ai.git

cd medvision-ai

pip install -r requirements.txt
```

---

## ▶️ Running the Project

Start Backend:

```bash
python backend/main.py
```

Start Model Service:

```bash
python model_service/main.py
```

Launch Frontend:

Open:

```text
frontend/index.html
```

---

## 🧪 Testing

```bash
pytest tests/
```

---

## 🔬 Future Enhancements

* LLM-powered clinical reasoning
* RAG-based medical knowledge retrieval
* Multi-modal report generation
* Explainable AI (XAI)
* Cloud deployment
* Real-time hospital integration

---

## 👨‍💻 Author

Mohd Mudabbir Arafat

B.Tech (Artificial Intelligence & Machine Learning)

GitHub:
https://github.com/Mudabbirarafat

---

## ⚠️ Disclaimer

This project is intended for educational and research purposes only and should not be used as a substitute for professional medical diagnosis.

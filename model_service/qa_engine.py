"""
model_service/qa_engine.py
---------------------------
Keyword-based clinical QA over model predictions.
Maps natural language questions to structured answers about detected conditions.
"""

import re
from typing import List, Dict

from training.dataset import SELECTED_LABELS

CONDITION_KEYWORDS = {
    "Atelectasis":      ["atelectasis", "collapse", "collapsed", "airway"],
    "Cardiomegaly":     ["cardiomegaly", "heart", "cardiac", "enlarged heart", "heart size"],
    "Consolidation":    ["consolidation", "consolidate", "infiltrate"],
    "Edema":            ["edema", "oedema", "fluid overload", "congestion", "wet lung"],
    "Pleural Effusion": ["pleural effusion", "effusion", "pleural fluid", "pleural"],
    "Pneumonia":        ["pneumonia", "pneumonic", "infection", "pna", "pneumonitis"],
}


def answer_question(question: str, predictions: List[Dict]) -> Dict:
    q = question.lower().strip().rstrip("?.!")
    lookup = {p["label"]: p for p in predictions}
    positive = [p for p in predictions if p["positive"]]

    # ── Is it normal? ───────────────────────────────────────────────────────────
    if any(w in q for w in ["normal", "clear", "healthy", "okay", "fine", "negative"]):
        if not positive:
            ans = "The AI analysis found NO significant abnormalities. The chest X-ray appears within normal limits. Always verify with a radiologist."
            conf = 0.85
        else:
            names = ", ".join(p["label"] for p in positive)
            ans = f"The X-ray is NOT normal. The following conditions were detected: {names}. Radiologist review is required."
            conf = max(p["probability"] for p in positive)
        return _response(question, ans, conf, [])

    # ── Specific condition yes/no ────────────────────────────────────────────────
    for label, keywords in CONDITION_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            pred = lookup.get(label)
            if pred:
                if pred["positive"]:
                    ans = (f"YES — {label} is detected with {pred['confidence_pct']}% confidence "
                           f"({pred['severity']} severity). Clinical correlation recommended.")
                else:
                    ans = (f"NO — {label} was NOT detected "
                           f"(probability {pred['confidence_pct']}%, below threshold). "
                           f"Cannot fully exclude; radiologist review advised.")
                return _response(question, ans, pred["probability"], [label])

    # ── Most likely / primary diagnosis ─────────────────────────────────────────
    if any(w in q for w in ["most likely", "primary", "main", "diagnosis", "top finding", "worst"]):
        if not positive:
            ans = "No significant finding was detected. The study appears normal."
        else:
            top = sorted(positive, key=lambda x: x["probability"], reverse=True)[0]
            ans = (f"The most likely finding is {top['label']} "
                   f"(confidence: {top['confidence_pct']}%, severity: {top['severity']}). "
                   f"{'Additional findings: ' + ', '.join(p['label'] for p in positive[1:]) + '.' if len(positive) > 1 else ''}")
        return _response(question, ans, max((p["probability"] for p in positive), default=0.5), [])

    # ── List all findings ────────────────────────────────────────────────────────
    if any(w in q for w in ["list", "all", "findings", "what", "show", "see", "found"]):
        if not positive:
            ans = "No significant conditions were detected. The study appears within normal limits."
        else:
            items = "\n".join(f"  • {p['label']} — {p['confidence_pct']}% confidence ({p['severity']} severity)"
                               for p in sorted(positive, key=lambda x: x["probability"], reverse=True))
            ans = f"The following conditions were detected ({len(positive)} finding{'s' if len(positive)>1 else ''}):\n{items}"
        conf = max((p["probability"] for p in positive), default=0.5)
        return _response(question, ans, conf, [p["label"] for p in positive])

    # ── Confidence / certainty ───────────────────────────────────────────────────
    if any(w in q for w in ["confident", "sure", "certainty", "accuracy", "reliable"]):
        top = sorted(predictions, key=lambda x: x["probability"], reverse=True)[0]
        ans = (f"The model's highest confidence finding is {top['label']} at {top['confidence_pct']}%. "
               f"Mean confidence across all 6 labels is "
               f"{int(sum(p['probability'] for p in predictions)/len(predictions)*100)}%. "
               f"Results should always be confirmed by a radiologist.")
        return _response(question, ans, top["probability"], [top["label"]])

    # ── Generic fallback ─────────────────────────────────────────────────────────
    if positive:
        names = ", ".join(p["label"] for p in sorted(positive, key=lambda x: x["probability"], reverse=True)[:3])
        ans = (f"Based on AI analysis, the following finding(s) were detected: {names}. "
               f"For a specific question, try asking: 'Is there pneumonia?' or 'What are the main findings?'")
    else:
        ans = "No significant findings were detected in this chest X-ray. The study appears normal."

    return _response(question, ans, 0.6, [p["label"] for p in positive])


def _response(question: str, answer: str, confidence: float, findings: List[str]) -> Dict:
    return {
        "question":          question,
        "answer":            answer,
        "confidence":        round(float(confidence), 4),
        "relevant_findings": findings,
        "disclaimer":        "AI-generated answer for educational purposes. Must be verified by a clinician.",
    }

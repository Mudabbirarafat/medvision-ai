"""
model_service/report_generator.py
-----------------------------------
Generates structured diagnostic reports from model predictions.
For a final-year project, this is rule-based + template NLG.
In production, this would use a fine-tuned clinical LLM.
"""

from typing import List, Dict, Tuple, Optional

# ─── Clinical knowledge per condition ──────────────────────────────────────────

FINDINGS_DB = {
    "Atelectasis": {
        "description": "Subsegmental or lobar atelectasis is present, consistent with airway collapse or mucus plugging.",
        "severity_text": {
            "high":     "Significant atelectasis with likely lobar collapse.",
            "moderate": "Moderate atelectasis, possibly subsegmental.",
            "low":      "Subtle linear atelectasis noted.",
        },
        "recommendation": "Incentive spirometry, chest physiotherapy. Re-imaging if clinically indicated.",
        "icd10": "J98.11",
    },
    "Cardiomegaly": {
        "description": "The cardiac silhouette is enlarged (cardiothoracic ratio >0.5), suggestive of cardiomegaly.",
        "severity_text": {
            "high":     "Marked cardiomegaly with CTR significantly elevated.",
            "moderate": "Moderate cardiomegaly noted.",
            "low":      "Mild cardiomegaly cannot be excluded.",
        },
        "recommendation": "Echocardiography recommended to assess ventricular function and valvular pathology. Cardiology referral.",
        "icd10": "I51.7",
    },
    "Consolidation": {
        "description": "Airspace consolidation is identified, raising concern for pneumonia or other acute pulmonary process.",
        "severity_text": {
            "high":     "Dense consolidation consistent with lobar or segmental pneumonia.",
            "moderate": "Patchy consolidation in keeping with acute infectious process.",
            "low":      "Subtle consolidation, early infectious or aspiration aetiology possible.",
        },
        "recommendation": "Correlation with inflammatory markers (WBC, CRP, procalcitonin). Sputum culture. Antibiotic therapy if confirmed. Repeat CXR in 4-6 weeks.",
        "icd10": "J18.9",
    },
    "Edema": {
        "description": "Perihilar haziness and increased vascular markings consistent with pulmonary oedema.",
        "severity_text": {
            "high":     "Florid pulmonary oedema with bilateral perihilar bat-wing opacification.",
            "moderate": "Moderate pulmonary oedema with interstitial changes.",
            "low":      "Early pulmonary oedema pattern, Kerley B lines may be present.",
        },
        "recommendation": "Diuretic therapy (furosemide). Fluid restriction. Cardiac biomarkers (BNP, troponin). Cardiology consultation.",
        "icd10": "J81.1",
    },
    "Pleural Effusion": {
        "description": "Pleural effusion is present, evidenced by blunting of the costophrenic angle.",
        "severity_text": {
            "high":     "Large pleural effusion causing significant volume loss.",
            "moderate": "Moderate pleural effusion with meniscus sign.",
            "low":      "Small pleural effusion noted.",
        },
        "recommendation": "Differentiate transudative vs exudative aetiology. Echocardiography if cardiac cause suspected. Thoracentesis if large or symptomatic.",
        "icd10": "J90",
    },
    "Pneumonia": {
        "description": "Findings are consistent with pneumonic consolidation.",
        "severity_text": {
            "high":     "Extensive pneumonic infiltrates; multilobar involvement possible.",
            "moderate": "Segmental pneumonia with opacity in the affected lobe.",
            "low":      "Early or resolving pneumonia pattern.",
        },
        "recommendation": "Antibiotic therapy per local guidelines. Blood cultures if febrile. CXR follow-up in 4-6 weeks to confirm resolution. Consider CT if atypical.",
        "icd10": "J18.9",
    },
}


def generate_report(
    predictions: List[Dict],
    positive_findings: List[str],
    patient_id: Optional[str] = None,
    clinical_context: Optional[str] = None,
    modality: str = "Chest X-Ray (PA view)",
) -> Tuple[Dict, str]:
    """
    Generate a structured diagnostic report.

    Args:
        predictions      : Full list of {label, probability, positive, severity}
        positive_findings: List of positive label names
        patient_id       : Optional patient ID string
        clinical_context : Optional clinical notes / indication
        modality         : Imaging modality string

    Returns:
        (report_dict, report_text)
    """
    indication = clinical_context or "Chest radiograph requested for clinical evaluation."
    technique = (
        f"{modality} acquired and analysed using MedVision AI (ResNet18 encoder, "
        f"multi-label sigmoid classifier trained on CheXpert dataset, "
        f"6-label subset: Atelectasis, Cardiomegaly, Consolidation, Edema, "
        f"Pleural Effusion, Pneumonia)."
    )

    # ── Findings section ────────────────────────────────────────────────────────
    if not positive_findings:
        findings = (
            "The lungs are clear bilaterally with no focal consolidation, effusion, or "
            "pneumothorax identified. The cardiac silhouette is within normal limits. "
            "The mediastinal contours appear unremarkable. No acute cardiopulmonary "
            "abnormality is detected by the AI classifier."
        )
    else:
        finding_lines = []
        for i, pred in enumerate([p for p in predictions if p["positive"]]):
            db = FINDINGS_DB.get(pred["label"], {})
            sev_text = db.get("severity_text", {}).get(pred["severity"], db.get("description", "Abnormality identified."))
            finding_lines.append(
                f"  {i+1}. {pred['label']} (confidence: {pred['confidence_pct']}%) — {sev_text}"
            )
        findings = "\n".join(finding_lines)

    # ── Impression section ──────────────────────────────────────────────────────
    if not positive_findings:
        impression = "No acute cardiopulmonary pathology detected by AI analysis. Normal chest radiograph."
    elif len(positive_findings) == 1:
        pred = next(p for p in predictions if p["label"] == positive_findings[0])
        impression = (
            f"1. {positive_findings[0]} — detected with {pred['confidence_pct']}% confidence. "
            f"Clinical correlation recommended."
        )
    else:
        lines = [
            f"  {i+1}. {name} — {next(p['confidence_pct'] for p in predictions if p['label']==name)}% confidence"
            for i, name in enumerate(positive_findings)
        ]
        impression = "\n".join(lines)
        impression += "\n  Clinical correlation with symptoms and laboratory findings strongly recommended."

    # ── Recommendations section ─────────────────────────────────────────────────
    recs = []
    for name in positive_findings:
        db = FINDINGS_DB.get(name, {})
        rec = db.get("recommendation")
        if rec:
            recs.append(f"• {name}: {rec}")
    if not recs:
        recs.append("• No specific intervention indicated based on AI analysis.")
    recs.append("• ⚠️ All AI findings must be reviewed and confirmed by a licensed radiologist.")
    recs.append("• This report is generated for educational/research purposes only.")

    recommendations = "\n".join(recs)

    disclaimer = (
        "DISCLAIMER: This report is generated by an AI system (MedVision v2.0) trained on "
        "the CheXpert dataset. It is intended for educational and research purposes only. "
        "It is NOT a substitute for professional radiological or clinical diagnosis. "
        "All findings must be verified by a qualified radiologist before any clinical "
        "decision is made. The system has not been approved by any regulatory authority."
    )

    report_dict = {
        "patient_id":       patient_id or "Anonymous",
        "modality":         modality,
        "indication":       indication,
        "technique":        technique,
        "findings":         findings,
        "impression":       impression,
        "recommendations":  recommendations,
        "positive_count":   len(positive_findings),
        "top_finding":      positive_findings[0] if positive_findings else "No Finding",
        "disclaimer":       disclaimer,
    }

    report_text = _format_text(report_dict)
    return report_dict, report_text


def _format_text(r: Dict) -> str:
    lines = [
        "=" * 62,
        "   MEDVISION AI — DIAGNOSTIC REPORT",
        "=" * 62,
        f"Patient ID     : {r['patient_id']}",
        f"Modality       : {r['modality']}",
        f"Indication     : {r['indication']}",
        "",
        "TECHNIQUE",
        "-" * 40,
        r["technique"],
        "",
        "FINDINGS",
        "-" * 40,
        r["findings"],
        "",
        "IMPRESSION",
        "-" * 40,
        r["impression"],
        "",
        "RECOMMENDATIONS",
        "-" * 40,
        r["recommendations"],
        "",
        "=" * 62,
        r["disclaimer"],
        "=" * 62,
    ]
    return "\n".join(lines)

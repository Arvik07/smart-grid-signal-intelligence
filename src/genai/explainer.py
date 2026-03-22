"""
Layer 5 — Generative AI
explainer.py

Generates natural language fault explanations and severity assessments
using LangChain + Groq (llama3-70b-8192, free tier).
"""

import os
import json
import re
from config import GROQ_API_KEY, GROQ_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from src.genai.prompts import (
    FAULT_EXPLANATION_PROMPT,
    SEVERITY_ASSESSMENT_PROMPT,
    DASHBOARD_SUMMARY_PROMPT,
)


# ── LLM initialisation ────────────────────────────────────────────────────

def get_llm(
    model: str = GROQ_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
):
    """
    Initialise and return the Groq LLM via LangChain.

    Args:
        model:       Groq model string
        temperature: sampling temperature (0 = deterministic)
        max_tokens:  max output tokens

    Returns:
        LangChain ChatGroq instance
    """
    from langchain_groq import ChatGroq

    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Add it to your .env file.\n"
            "Get a free key at: https://console.groq.com"
        )

    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        groq_api_key=GROQ_API_KEY,
    )


# ── Core chain builder ────────────────────────────────────────────────────

def build_chain(prompt_template):
    """
    Build a simple LangChain chain: prompt → LLM → string output.

    Args:
        prompt_template: LangChain PromptTemplate

    Returns:
        Runnable chain
    """
    from langchain_core.output_parsers import StrOutputParser

    llm   = get_llm()
    chain = prompt_template | llm | StrOutputParser()
    return chain


# ── Fault explanation ─────────────────────────────────────────────────────

def explain_fault(
    fault_name: str,
    confidence: float,
    features: dict,
    fft_data: dict,
) -> str:
    """
    Generate a natural language explanation of the detected fault.

    Args:
        fault_name:  detected fault class name (e.g. 'harmonic_distortion')
        confidence:  classifier confidence 0-100
        features:    feature dict from extract_features()
        fft_data:    fft_summary() dict

    Returns:
        explanation string
    """
    chain = build_chain(FAULT_EXPLANATION_PROMPT)

    inputs = {
        "fault_name":       fault_name.replace("_", " ").title(),
        "confidence":       round(confidence * 100, 1) if confidence <= 1.0 else round(confidence, 1),
        "rms":              features.get("rms", 0.0),
        "peak_value":       features.get("peak_value", 0.0),
        "crest_factor":     features.get("crest_factor", 0.0),
        "thd_percent":      features.get("thd_percent", 0.0),
        "dominant_freq":    fft_data.get("dominant_freq", 50.0),
        "freq_deviation_hz": features.get("freq_deviation_hz", 0.0),
        "spectral_entropy": features.get("spectral_entropy", 0.0),
        "kurtosis":         features.get("kurtosis", 3.0),
        "waveform_deviation": features.get("waveform_deviation", 0.0),
        "ihd_3":            features.get("ihd_3", 0.0),
        "ihd_5":            features.get("ihd_5", 0.0),
        "ihd_7":            features.get("ihd_7", 0.0),
        "ihd_9":            features.get("ihd_9", 0.0),
        "ihd_11":           features.get("ihd_11", 0.0),
    }

    try:
        return chain.invoke(inputs).strip()
    except Exception as e:
        return f"[Explanation unavailable: {str(e)}]"


# ── Severity assessment ───────────────────────────────────────────────────

def assess_severity(
    fault_name: str,
    features: dict,
    anomaly_score: float = None,
) -> dict:
    """
    Ask the LLM to classify fault severity as LOW/MEDIUM/HIGH/CRITICAL.

    Args:
        fault_name:    detected fault class name
        features:      feature dict from extract_features()
        anomaly_score: Isolation Forest score (optional)

    Returns:
        dict with keys: severity, reason
    """
    chain = build_chain(SEVERITY_ASSESSMENT_PROMPT)

    inputs = {
        "fault_name":         fault_name.replace("_", " ").title(),
        "thd_percent":        features.get("thd_percent", 0.0),
        "freq_deviation_hz":  features.get("freq_deviation_hz", 0.0),
        "crest_factor":       features.get("crest_factor", 0.0),
        "waveform_deviation": features.get("waveform_deviation", 0.0),
        "anomaly_score":      str(round(anomaly_score, 4)) if anomaly_score else "N/A",
    }

    try:
        raw = chain.invoke(inputs).strip()

        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip("` \n")

        result = json.loads(raw)
        severity = result.get("severity", "MEDIUM").upper()
        reason   = result.get("reason", "")

        # Validate
        valid = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        if severity not in valid:
            severity = "MEDIUM"

        return {"severity": severity, "reason": reason}

    except (json.JSONDecodeError, Exception) as e:
        # Fallback: rule-based severity
        return _rule_based_severity(features)


def _rule_based_severity(features: dict) -> dict:
    """
    Fallback rule-based severity when LLM is unavailable.
    Mirrors the rules in the severity assessment prompt.
    """
    thd   = features.get("thd_percent", 0.0)
    cf    = features.get("crest_factor", 0.0)
    fdev  = abs(features.get("freq_deviation_hz", 0.0))

    if thd > 20.0 or cf > 3.0:
        return {"severity": "HIGH", "reason": "THD or crest factor exceeds safe limits."}
    elif thd > 5.0 or fdev > 2.0:
        return {"severity": "MEDIUM", "reason": "Signal quality below IEEE standards."}
    elif thd > 2.0 or fdev > 0.5:
        return {"severity": "LOW", "reason": "Minor signal deviation detected."}
    else:
        return {"severity": "LOW", "reason": "Signal within acceptable parameters."}


# ── Dashboard summary ─────────────────────────────────────────────────────

def generate_dashboard_summary(
    fault_name: str,
    severity: str,
    confidence: float,
    features: dict,
    fft_data: dict,
    is_anomaly: bool,
) -> str:
    """
    Generate a short plain-language status summary for the dashboard header.

    Args:
        fault_name:  detected fault class name
        severity:    severity string (LOW/MEDIUM/HIGH/CRITICAL)
        confidence:  classifier confidence 0-1
        features:    feature dict
        fft_data:    fft_summary() dict
        is_anomaly:  whether anomaly detector flagged this signal

    Returns:
        short summary string (≤60 words)
    """
    chain = build_chain(DASHBOARD_SUMMARY_PROMPT)

    inputs = {
        "fault_name":   fault_name.replace("_", " ").title(),
        "severity":     severity,
        "confidence":   round(confidence * 100, 1) if confidence <= 1.0 else round(confidence, 1),
        "thd_percent":  features.get("thd_percent", 0.0),
        "dominant_freq": fft_data.get("dominant_freq", 50.0),
        "is_anomaly":   "Yes" if is_anomaly else "No",
    }

    try:
        return chain.invoke(inputs).strip()
    except Exception as e:
        fault_display = fault_name.replace("_", " ").title()
        return (
            f"{severity} severity {fault_display} detected with "
            f"{inputs['confidence']}% confidence. "
            f"THD: {inputs['thd_percent']:.1f}%. Immediate inspection recommended."
        )


# ── Full diagnostic pipeline ──────────────────────────────────────────────

def run_full_diagnosis(
    fault_name: str,
    confidence: float,
    features: dict,
    fft_data: dict,
    anomaly_score: float = None,
    is_anomaly: bool = False,
) -> dict:
    """
    Run the complete GenAI diagnosis pipeline in one call.
    Returns explanation, severity, corrective actions, and dashboard summary.

    Args:
        fault_name:    detected fault class name
        confidence:    classifier confidence (0-1 or 0-100)
        features:      feature dict from extract_features()
        fft_data:      fft_summary() dict
        anomaly_score: Isolation Forest score (optional)
        is_anomaly:    whether anomaly detector flagged the signal

    Returns:
        dict with keys: explanation, severity, reason, summary
    """
    print(f"\n[GenAI] Diagnosing: {fault_name} (confidence={confidence:.2%})")

    # 1. Severity
    print("[GenAI] Assessing severity...")
    sev_result  = assess_severity(fault_name, features, anomaly_score)
    severity    = sev_result["severity"]
    sev_reason  = sev_result["reason"]

    # 2. Explanation
    print("[GenAI] Generating explanation...")
    explanation = explain_fault(fault_name, confidence, features, fft_data)

    # 3. Dashboard summary
    print("[GenAI] Generating dashboard summary...")
    summary = generate_dashboard_summary(
        fault_name, severity, confidence, features, fft_data, is_anomaly
    )

    print(f"[GenAI] Done — severity={severity}")

    return {
        "fault_name":   fault_name,
        "severity":     severity,
        "reason":       sev_reason,
        "explanation":  explanation,
        "summary":      summary,
        "confidence":   confidence,
        "is_anomaly":   is_anomaly,
    }
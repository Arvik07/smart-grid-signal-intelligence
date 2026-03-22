"""
Layer 5 — Generative AI
recommender.py

Generates prioritised corrective action recommendations using
LangChain + Groq. Parses structured output from the LLM into
actionable items for the dashboard and reports.
"""

import re
from config import GROQ_MODEL, LLM_TEMPERATURE
from src.genai.prompts import CORRECTIVE_ACTION_PROMPT
from src.genai.explainer import build_chain


# ── Fault-to-actions knowledge base (fallback) ────────────────────────────

_FALLBACK_ACTIONS = {
    "normal": [
        ("IMMEDIATE",   "Continue standard monitoring — no corrective action required."),
        ("SHORT-TERM",  "Schedule next routine power quality audit per IEC 61000-4-30."),
        ("LONG-TERM",   "Maintain predictive maintenance schedule for all connected equipment."),
    ],
    "harmonic_distortion": [
        ("IMMEDIATE",   "Install passive harmonic filters (5th and 7th order) at the PCC."),
        ("SHORT-TERM",  "Audit all non-linear loads (VFDs, UPS, rectifiers) and quantify harmonic injection."),
        ("LONG-TERM",   "Deploy active power filters or upgrade to 12-pulse rectifier topology to maintain THD < 5% per IEEE 519."),
    ],
    "voltage_sag": [
        ("IMMEDIATE",   "Switch to UPS or automatic voltage regulator (AVR) to protect sensitive loads."),
        ("SHORT-TERM",  "Install Dynamic Voltage Restorer (DVR) or Static VAr Compensator (SVC) at the feeder."),
        ("LONG-TERM",   "Upgrade transformer tap settings and reinforce distribution feeder impedance."),
    ],
    "voltage_swell": [
        ("IMMEDIATE",   "Activate surge protection devices (SPDs) and check capacitor bank switching."),
        ("SHORT-TERM",  "Install voltage limiters and review reactive power compensation settings."),
        ("LONG-TERM",   "Deploy Static Synchronous Compensator (STATCOM) for continuous voltage regulation."),
    ],
    "transient_spike": [
        ("IMMEDIATE",   "Engage transient voltage surge suppressor (TVSS) and isolate the affected feeder."),
        ("SHORT-TERM",  "Install metal oxide varistors (MOVs) and snubber circuits at load terminals."),
        ("LONG-TERM",   "Implement lightning arrestors and grounding improvements per IEEE 1100."),
    ],
    "frequency_deviation": [
        ("IMMEDIATE",   "Activate automatic generation control (AGC) to rebalance generation-load mismatch."),
        ("SHORT-TERM",  "Check governor response on synchronous generators and adjust droop settings."),
        ("LONG-TERM",   "Integrate battery energy storage system (BESS) for primary frequency response under IEEE 1547."),
    ],
}


# ── LLM-based recommender ─────────────────────────────────────────────────

def get_corrective_actions(
    fault_name: str,
    severity: str,
    features: dict,
    fault_explanation: str,
    use_llm: bool = True,
) -> list:
    """
    Generate 3 prioritised corrective actions for a detected fault.

    Args:
        fault_name:        detected fault class name
        severity:          severity string (LOW/MEDIUM/HIGH/CRITICAL)
        features:          feature dict from extract_features()
        fault_explanation: explanation string from explainer.explain_fault()
        use_llm:           use LLM (True) or fallback knowledge base (False)

    Returns:
        list of dicts: [{"priority": str, "action": str}, ...]
    """
    if use_llm:
        try:
            return _llm_corrective_actions(
                fault_name, severity, features, fault_explanation
            )
        except Exception as e:
            print(f"[Recommender] LLM failed ({e}), using fallback.")

    return _fallback_corrective_actions(fault_name)


def _llm_corrective_actions(
    fault_name: str,
    severity: str,
    features: dict,
    fault_explanation: str,
) -> list:
    """
    Use LLM to generate corrective actions and parse them into structured list.
    """
    chain = build_chain(CORRECTIVE_ACTION_PROMPT)

    inputs = {
        "fault_name":       fault_name.replace("_", " ").title(),
        "severity":         severity,
        "thd_percent":      features.get("thd_percent", 0.0),
        "freq_deviation_hz": features.get("freq_deviation_hz", 0.0),
        "crest_factor":     features.get("crest_factor", 0.0),
        "fault_explanation": fault_explanation,
    }

    raw = chain.invoke(inputs).strip()
    return _parse_actions(raw)


def _parse_actions(raw_text: str) -> list:
    """
    Parse LLM output into structured action list.

    Expected format:
        ACTION 1 [IMMEDIATE]: <action text>
        ACTION 2 [SHORT-TERM]: <action text>
        ACTION 3 [LONG-TERM]: <action text>

    Returns:
        list of dicts: [{"priority": ..., "action": ...}, ...]
    """
    actions = []

    # Match "ACTION N [PRIORITY]: text"
    pattern = re.compile(
        r"ACTION\s+\d+\s+\[([^\]]+)\]\s*:\s*(.+)",
        re.IGNORECASE
    )
    matches = pattern.findall(raw_text)

    if matches:
        for priority, action in matches:
            actions.append({
                "priority": priority.strip().upper(),
                "action":   action.strip(),
            })
        return actions

    # Fallback: split by newlines and return raw lines
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
    priorities = ["IMMEDIATE", "SHORT-TERM", "LONG-TERM"]
    for i, line in enumerate(lines[:3]):
        actions.append({
            "priority": priorities[i] if i < len(priorities) else f"ACTION {i+1}",
            "action":   line,
        })

    return actions


def _fallback_corrective_actions(fault_name: str) -> list:
    """
    Return hardcoded expert actions when LLM is unavailable.
    """
    fault_key = fault_name.lower().replace(" ", "_")
    raw_actions = _FALLBACK_ACTIONS.get(fault_key, _FALLBACK_ACTIONS["normal"])
    return [
        {"priority": p, "action": a}
        for p, a in raw_actions
    ]


# ── Severity-aware action filter ──────────────────────────────────────────

def get_urgent_actions(actions: list, severity: str) -> list:
    """
    Filter and return only the most urgent actions based on severity.

    Args:
        actions:  list of action dicts from get_corrective_actions()
        severity: severity string (LOW/MEDIUM/HIGH/CRITICAL)

    Returns:
        filtered list — CRITICAL returns all 3, LOW returns only IMMEDIATE
    """
    if severity == "CRITICAL":
        return actions                   # all 3 actions
    elif severity == "HIGH":
        return actions[:2]               # IMMEDIATE + SHORT-TERM
    elif severity == "MEDIUM":
        return actions[:2]               # IMMEDIATE + SHORT-TERM
    else:
        return actions[:1]               # IMMEDIATE only


# ── Report builder ────────────────────────────────────────────────────────

def build_diagnostic_report(
    fault_name: str,
    severity: str,
    confidence: float,
    explanation: str,
    actions: list,
    features: dict,
    fft_data: dict,
) -> str:
    """
    Build a formatted plain-text diagnostic report combining all outputs.
    Suitable for logging, export, or display in the dashboard.

    Returns:
        formatted report string
    """
    conf_pct = round(confidence * 100, 1) if confidence <= 1.0 else round(confidence, 1)
    fault_display = fault_name.replace("_", " ").upper()

    separator = "=" * 60
    thin_sep  = "-" * 60

    report_lines = [
        separator,
        "  SMART GRID SIGNAL INTELLIGENCE — DIAGNOSTIC REPORT",
        separator,
        f"  Fault Type  : {fault_display}",
        f"  Severity    : {severity}",
        f"  Confidence  : {conf_pct}%",
        thin_sep,
        "  KEY MEASUREMENTS",
        thin_sep,
        f"  RMS                 : {features.get('rms', 0):.4f}",
        f"  Crest Factor        : {features.get('crest_factor', 0):.4f}  (nominal 1.414)",
        f"  THD                 : {features.get('thd_percent', 0):.2f}%  (IEEE limit 5%)",
        f"  Frequency           : {fft_data.get('dominant_freq', 50):.3f} Hz",
        f"  Frequency Deviation : {features.get('freq_deviation_hz', 0):.3f} Hz",
        f"  Spectral Entropy    : {features.get('spectral_entropy', 0):.4f}",
        f"  Waveform Deviation  : {features.get('waveform_deviation', 0):.4f}",
        thin_sep,
        "  HARMONIC PROFILE",
        thin_sep,
        f"  3rd (150 Hz) : {features.get('ihd_3', 0):.2f}%",
        f"  5th (250 Hz) : {features.get('ihd_5', 0):.2f}%",
        f"  7th (350 Hz) : {features.get('ihd_7', 0):.2f}%",
        f"  9th (450 Hz) : {features.get('ihd_9', 0):.2f}%",
        f" 11th (550 Hz) : {features.get('ihd_11', 0):.2f}%",
        thin_sep,
        "  FAULT EXPLANATION",
        thin_sep,
        f"  {explanation}",
        thin_sep,
        "  CORRECTIVE ACTIONS",
        thin_sep,
    ]

    for i, act in enumerate(actions, 1):
        report_lines.append(f"  {i}. [{act['priority']}] {act['action']}")

    report_lines.append(separator)

    return "\n".join(report_lines)
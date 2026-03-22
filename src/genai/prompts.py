"""
Layer 5 — Generative AI
prompts.py

All LLM prompt templates for the Smart Grid Signal Intelligence system.
Centralised here so prompts can be tuned without touching logic files.
Uses LangChain PromptTemplate for clean variable injection.
"""

from langchain_core.prompts import PromptTemplate


# ── Fault explanation prompt ───────────────────────────────────────────────

FAULT_EXPLANATION_TEMPLATE = """You are an expert power systems engineer specialising in power quality analysis and smart grid diagnostics.

A signal analysis system has detected the following condition in a 50 Hz power grid signal:

DETECTED FAULT: {fault_name}
CONFIDENCE: {confidence}%

SIGNAL MEASUREMENTS:
- RMS Voltage (normalised): {rms:.4f}
- Peak Value: {peak_value:.4f}
- Crest Factor: {crest_factor:.4f}  (normal = 1.414 for pure sine)
- THD (Total Harmonic Distortion): {thd_percent:.2f}%  (IEEE limit = 5%)
- Dominant Frequency: {dominant_freq:.2f} Hz  (nominal = 50 Hz)
- Frequency Deviation: {freq_deviation_hz:.3f} Hz
- Spectral Entropy: {spectral_entropy:.4f}
- Kurtosis: {kurtosis:.4f}  (normal ≈ 3.0)
- Waveform Deviation: {waveform_deviation:.4f}

HARMONIC PROFILE:
- 3rd harmonic (150 Hz): {ihd_3:.2f}%
- 5th harmonic (250 Hz): {ihd_5:.2f}%
- 7th harmonic (350 Hz): {ihd_7:.2f}%
- 9th harmonic (450 Hz): {ihd_9:.2f}%
- 11th harmonic (550 Hz): {ihd_11:.2f}%

Provide a concise but technically precise explanation of:
1. What this fault means physically in the power system
2. What is likely causing it (common real-world sources)
3. What risks it poses if left unaddressed (equipment, safety, efficiency)

Keep your response under 200 words. Be specific — reference the actual measured values above.
Do not use bullet points. Write in clear technical prose.
"""

FAULT_EXPLANATION_PROMPT = PromptTemplate(
    input_variables=[
        "fault_name", "confidence",
        "rms", "peak_value", "crest_factor",
        "thd_percent", "dominant_freq", "freq_deviation_hz",
        "spectral_entropy", "kurtosis", "waveform_deviation",
        "ihd_3", "ihd_5", "ihd_7", "ihd_9", "ihd_11",
    ],
    template=FAULT_EXPLANATION_TEMPLATE,
)


# ── Corrective action prompt ───────────────────────────────────────────────

CORRECTIVE_ACTION_TEMPLATE = """You are a senior power systems engineer providing actionable recommendations.

DETECTED FAULT: {fault_name}
SEVERITY: {severity}
THD: {thd_percent:.2f}% | Frequency Deviation: {freq_deviation_hz:.3f} Hz | Crest Factor: {crest_factor:.4f}

FAULT SUMMARY:
{fault_explanation}

Based on the above fault and its measured parameters, provide exactly 3 specific corrective actions.
Each action must be:
- Technically specific (name the actual equipment, standard, or technique)
- Ordered by priority (most urgent first)
- Realistic for an industrial smart grid environment

Format your response as:
ACTION 1 [IMMEDIATE]: <action>
ACTION 2 [SHORT-TERM]: <action>
ACTION 3 [LONG-TERM]: <action>

Be concise. One sentence per action. No extra commentary.
"""

CORRECTIVE_ACTION_PROMPT = PromptTemplate(
    input_variables=[
        "fault_name", "severity",
        "thd_percent", "freq_deviation_hz", "crest_factor",
        "fault_explanation",
    ],
    template=CORRECTIVE_ACTION_TEMPLATE,
)


# ── Severity assessment prompt ─────────────────────────────────────────────

SEVERITY_ASSESSMENT_TEMPLATE = """You are a power quality analyst. Assess the severity of this detected grid fault.

FAULT TYPE: {fault_name}
THD: {thd_percent:.2f}%
FREQUENCY DEVIATION: {freq_deviation_hz:.3f} Hz
CREST FACTOR: {crest_factor:.4f}
WAVEFORM DEVIATION: {waveform_deviation:.4f}
ANOMALY SCORE: {anomaly_score}

Classify the severity as exactly one of: LOW | MEDIUM | HIGH | CRITICAL

Rules:
- THD > 20% or crest factor > 3.0 → at least HIGH
- Frequency deviation > 2 Hz → at least MEDIUM
- THD > 5% (IEEE limit) → at least MEDIUM
- Multiple simultaneous anomalies → escalate by one level

Respond with ONLY this JSON (no markdown, no explanation):
{{"severity": "LOW|MEDIUM|HIGH|CRITICAL", "reason": "one sentence justification"}}
"""

SEVERITY_ASSESSMENT_PROMPT = PromptTemplate(
    input_variables=[
        "fault_name", "thd_percent", "freq_deviation_hz",
        "crest_factor", "waveform_deviation", "anomaly_score",
    ],
    template=SEVERITY_ASSESSMENT_TEMPLATE,
)


# ── Dashboard summary prompt ───────────────────────────────────────────────

DASHBOARD_SUMMARY_TEMPLATE = """You are a smart grid monitoring system. Generate a one-paragraph real-time status summary.

CURRENT GRID STATUS:
- Fault Detected: {fault_name}
- Severity: {severity}
- Confidence: {confidence}%
- THD: {thd_percent:.2f}%
- Frequency: {dominant_freq:.2f} Hz
- Anomaly Detected: {is_anomaly}

Write a single short paragraph (max 60 words) suitable for display on a monitoring dashboard.
Use plain language. State the fault, its severity, and the single most important action.
No technical jargon. No lists. Present tense.
"""

DASHBOARD_SUMMARY_PROMPT = PromptTemplate(
    input_variables=[
        "fault_name", "severity", "confidence",
        "thd_percent", "dominant_freq", "is_anomaly",
    ],
    template=DASHBOARD_SUMMARY_TEMPLATE,
)
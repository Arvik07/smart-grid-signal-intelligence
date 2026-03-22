"""
Layer 6 — Dashboard
components.py

Reusable Streamlit UI components — metric cards, status badges,
diagnostic panels, and sidebar controls.
"""

import streamlit as st
import numpy as np
from config import FAULT_TYPES

SEVERITY_EMOJI = {
    "LOW":      "🟢",
    "MEDIUM":   "🟡",
    "HIGH":     "🔴",
    "CRITICAL": "🚨",
}

FAULT_EMOJI = {
    "normal":               "✅",
    "harmonic_distortion":  "〰️",
    "voltage_sag":          "📉",
    "voltage_swell":        "📈",
    "transient_spike":      "⚡",
    "frequency_deviation":  "🔄",
}


# ── Page config ───────────────────────────────────────────────────────────

def set_page_config():
    """Set Streamlit page-level configuration. Call once at top of app.py."""
    st.set_page_config(
        page_title="Smart Grid Signal Intelligence",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_css():
    """Inject custom CSS for metric cards and status badges."""
    st.markdown("""
    <style>
    .metric-card {
        background: rgba(55, 138, 221, 0.08);
        border: 1px solid rgba(55, 138, 221, 0.25);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }
    .metric-label  { font-size: 12px; color: #888; margin-bottom: 2px; }
    .metric-value  { font-size: 22px; font-weight: 600; }
    .badge         { display: inline-block; padding: 3px 12px;
                     border-radius: 20px; font-size: 13px; font-weight: 600; }
    .badge-low      { background:#d4edda; color:#155724; }
    .badge-medium   { background:#fff3cd; color:#856404; }
    .badge-high     { background:#f8d7da; color:#721c24; }
    .badge-critical { background:#8E1A0E; color:#fff; }
    .report-box {
        background: rgba(0,0,0,0.04);
        border-left: 3px solid #378ADD;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        font-family: monospace;
        font-size: 12px;
        white-space: pre-wrap;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────

def render_header(summary: str = "", severity: str = "LOW"):
    """Top-of-page header with title and live status summary."""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.title("⚡ Smart Grid Signal Intelligence")
        if summary:
            sev_colour = {"LOW": "#2ECC71", "MEDIUM": "#F39C12",
                          "HIGH": "#E74C3C", "CRITICAL": "#8E1A0E"}.get(severity, "#888")
            st.markdown(
                f'<div style="border-left:4px solid {sev_colour}; '
                f'padding: 8px 14px; border-radius: 0 6px 6px 0;">'
                f'{summary}</div>',
                unsafe_allow_html=True
            )

    with col2:
        badge_class = f"badge-{severity.lower()}"
        emoji       = SEVERITY_EMOJI.get(severity, "")
        st.markdown(
            f'<div style="text-align:right; padding-top:18px;">'
            f'<span class="badge {badge_class}">{emoji} {severity}</span>'
            f'</div>',
            unsafe_allow_html=True
        )


# ── Metric cards ──────────────────────────────────────────────────────────

def render_signal_metrics(features: dict, fft_data: dict):
    """
    Row of key signal metric cards.
    Shows RMS, THD, Crest Factor, Dominant Frequency, Frequency Deviation.
    """
    cols = st.columns(5)

    metrics = [
        ("RMS",             f"{features.get('rms', 0):.4f}",          "p.u."),
        ("THD",             f"{features.get('thd_percent', 0):.2f}",   "%"),
        ("Crest Factor",    f"{features.get('crest_factor', 0):.4f}",  "(nom. 1.414)"),
        ("Frequency",       f"{fft_data.get('dominant_freq', 50):.3f}","Hz"),
        ("Freq Deviation",  f"{features.get('freq_deviation_hz', 0):.3f}", "Hz"),
    ]

    for col, (label, value, unit) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div>'
                f'<div class="metric-label">{unit}</div>'
                f'</div>',
                unsafe_allow_html=True
            )


def render_fault_card(
    fault_name: str,
    confidence: float,
    is_anomaly: bool,
    severity: str,
):
    """
    Prominent fault detection result card shown at the top of results.
    """
    emoji      = FAULT_EMOJI.get(fault_name, "⚠️")
    sev_emoji  = SEVERITY_EMOJI.get(severity, "")
    conf_pct   = round(confidence * 100, 1) if confidence <= 1.0 else round(confidence, 1)
    anomaly_str = "🔴 Anomaly detected" if is_anomaly else "🟢 No anomaly"
    fault_disp  = fault_name.replace("_", " ").title()

    st.markdown(
        f"""
        <div style="background:rgba(55,138,221,0.06); border:1px solid rgba(55,138,221,0.3);
                    border-radius:12px; padding:20px 24px; margin-bottom:16px;">
            <div style="font-size:28px; font-weight:700; margin-bottom:6px;">
                {emoji} {fault_disp}
            </div>
            <div style="display:flex; gap:20px; flex-wrap:wrap; font-size:14px; color:#888;">
                <span>Confidence: <b style="color:#eee">{conf_pct}%</b></span>
                <span>Severity: <b style="color:#eee">{sev_emoji} {severity}</b></span>
                <span>{anomaly_str}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ── Explanation panel ─────────────────────────────────────────────────────

def render_explanation(explanation: str, reason: str = ""):
    """Render GenAI fault explanation with styled container."""
    st.subheader("AI Fault Explanation")

    if reason:
        st.caption(f"Severity rationale: {reason}")

    st.markdown(
        f'<div style="background:rgba(0,0,0,0.04); border-left:3px solid #378ADD; '
        f'padding:14px 18px; border-radius:0 8px 8px 0; line-height:1.7;">'
        f'{explanation}'
        f'</div>',
        unsafe_allow_html=True
    )


# ── Corrective actions panel ──────────────────────────────────────────────

def render_corrective_actions(actions: list):
    """Render prioritised corrective actions as a styled list."""
    st.subheader("Corrective Actions")

    priority_colours = {
        "IMMEDIATE":  "#E74C3C",
        "SHORT-TERM": "#F39C12",
        "LONG-TERM":  "#2ECC71",
    }

    for i, act in enumerate(actions, 1):
        priority = act.get("priority", "ACTION")
        action   = act.get("action", "")
        colour   = priority_colours.get(priority, "#378ADD")

        st.markdown(
            f'<div style="display:flex; align-items:flex-start; gap:12px; '
            f'margin-bottom:10px; padding:10px 14px; '
            f'background:rgba(0,0,0,0.03); border-radius:8px;">'
            f'<span style="background:{colour}; color:white; padding:2px 10px; '
            f'border-radius:12px; font-size:11px; font-weight:600; white-space:nowrap;">'
            f'{priority}</span>'
            f'<span style="font-size:14px; line-height:1.5;">{action}</span>'
            f'</div>',
            unsafe_allow_html=True
        )


# ── Diagnostic report ─────────────────────────────────────────────────────

def render_report(report_text: str):
    """Render full diagnostic report in a monospace expandable box."""
    with st.expander("Full diagnostic report", expanded=False):
        st.markdown(
            f'<div class="report-box">{report_text}</div>',
            unsafe_allow_html=True
        )
        st.download_button(
            label="Download report (.txt)",
            data=report_text,
            file_name="smart_grid_diagnostic_report.txt",
            mime="text/plain",
        )


# ── Sidebar controls ──────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """
    Render the sidebar with all user controls.

    Returns:
        dict with selected options:
            fault_type, n_samples, add_noise, noise_snr,
            run_genai, show_spectrogram, show_radar
    """
    st.sidebar.title("⚙️ Signal Controls")

    st.sidebar.markdown("### Signal type")
    fault_names = [FAULT_TYPES[i] for i in sorted(FAULT_TYPES.keys())]
    selected_fault = st.sidebar.selectbox(
        "Fault type",
        options=fault_names,
        index=0,
        format_func=lambda x: x.replace("_", " ").title()
    )
    fault_type = [k for k, v in FAULT_TYPES.items() if v == selected_fault][0]

    st.sidebar.markdown("### Noise settings")
    add_noise = st.sidebar.checkbox("Add realistic noise", value=True)
    noise_snr = st.sidebar.slider(
        "SNR (dB)", min_value=10, max_value=50, value=30, step=5,
        disabled=not add_noise
    )

    st.sidebar.markdown("### Analysis options")
    run_genai        = st.sidebar.checkbox("Enable AI diagnostics (Groq)", value=True)
    show_spectrogram = st.sidebar.checkbox("Show spectrogram", value=True)
    show_radar       = st.sidebar.checkbox("Show feature radar", value=True)
    show_harmonics   = st.sidebar.checkbox("Show harmonic bars", value=True)

    st.sidebar.markdown("---")
    st.sidebar.caption("Smart Grid Signal Intelligence\nPowered by Groq + LangChain")

    return {
        "fault_type":       fault_type,
        "fault_name":       selected_fault,
        "add_noise":        add_noise,
        "noise_snr":        noise_snr,
        "run_genai":        run_genai,
        "show_spectrogram": show_spectrogram,
        "show_radar":       show_radar,
        "show_harmonics":   show_harmonics,
    }
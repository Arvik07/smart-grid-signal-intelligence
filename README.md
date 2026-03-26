# Smart Grid Signal Intelligence System

An end-to-end intelligent system that simulates power grid signals, detects anomalies using DSP + ML, predicts future faults with LSTM, and generates explainable AI diagnostics via Groq LLM — all visualised in a Streamlit dashboard.

Live Demo: smart-grid-signal-intelligence-kbg43qeueyjmrlnqkryv4e.streamlit.app/

## Stack
- **Signal simulation** — NumPy, SciPy
- **DSP** — SciPy (FFT, FIR/IIR filters, STFT)
- **ML** — Scikit-learn (Random Forest, Isolation Forest), TensorFlow (LSTM)
- **GenAI** — LangChain + Groq (llama3-70b, free)
- **Dashboard** — Streamlit + Plotly

## Setup
```bash
conda create -p ./env python=3.11 -y
conda activate ./env
pip install -r requirements.txt
```

Add your Groq API key to `.env`:
```
GROQ_API_KEY=your_key_here
```

## Run
```bash
streamlit run src/dashboard/app.py
```

## Project structure
```
src/
  simulation/   # Layer 1 — synthetic signal generation
  dsp/          # Layer 2 — FFT, filtering, spectrogram
  features/     # Layer 3 — THD, spectral entropy, RMS
  ml/           # Layer 4 — classifier, anomaly detector, LSTM
  genai/        # Layer 5 — LLM explanation & recommendations
  dashboard/    # Layer 6 — Streamlit UI
```

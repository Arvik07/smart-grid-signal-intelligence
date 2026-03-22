import os
from pathlib import Path
from dotenv import load_dotenv

# Explicitly resolve .env from the project root
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_RAW        = BASE_DIR / "data" / "raw"
DATA_PROCESSED  = BASE_DIR / "data" / "processed"
MODELS_DIR      = BASE_DIR / "data" / "models"

# ── Signal simulation ──────────────────────────────────────────────────────
SAMPLING_RATE   = 10_000        # Hz  (10 kHz)
SIGNAL_FREQ     = 50            # Hz  (50 Hz power grid)
SIGNAL_DURATION = 1.0           # seconds per sample
NUM_SAMPLES     = int(SAMPLING_RATE * SIGNAL_DURATION)
AMPLITUDE       = 1.0           # normalised

# Fault types (used as class labels)
FAULT_TYPES = {
    0: "normal",
    1: "harmonic_distortion",
    2: "voltage_sag",
    3: "voltage_swell",
    4: "transient_spike",
    5: "frequency_deviation",
}

# Harmonic orders to inject
HARMONIC_ORDERS = [3, 5, 7, 9, 11]

# ── Feature extraction ─────────────────────────────────────────────────────
WINDOW_SIZE     = 1000          # samples per feature window
WINDOW_STEP     = 500           # 50% overlap
FFT_BINS        = 512

# ── ML ─────────────────────────────────────────────────────────────────────
TEST_SIZE       = 0.2
RANDOM_STATE    = 42
LSTM_SEQUENCE_LEN = 50          # time steps fed into LSTM
LSTM_EPOCHS     = 30
LSTM_BATCH_SIZE = 32

# ── GenAI (Groq) ───────────────────────────────────────────────────────────
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GROQ_MODEL      = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS  = 512

# ── Dashboard ──────────────────────────────────────────────────────────────
DASHBOARD_TITLE = "Smart Grid Signal Intelligence"
REFRESH_INTERVAL = 2            # seconds
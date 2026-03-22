"""
pipeline.py

End-to-end runner for the Smart Grid Signal Intelligence system.
Executes all 6 layers in sequence:

  Layer 1 — Signal simulation
  Layer 2 — DSP analysis
  Layer 3 — Feature extraction
  Layer 4 — ML training (Random Forest + Isolation Forest + LSTM)
  Layer 5 — GenAI diagnosis (Groq)
  Layer 6 — Report generation

Usage:
    python pipeline.py                                  # full run, harmonic_distortion
    python pipeline.py --fault voltage_sag              # specific fault type
    python pipeline.py --fault transient_spike --skip-training
    python pipeline.py --n-per-class 300 --no-lstm      # skip LSTM (faster)
    python pipeline.py --all-faults                     # diagnose all 6 fault types
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

# ── Ensure project root is importable ────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import FAULT_TYPES


# ── Helpers ───────────────────────────────────────────────────────────────

class Timer:
    """Simple context manager for timing steps."""
    def __init__(self, label: str):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"  ✓ {self.label} completed in {self.elapsed:.1f}s")


def section(title: str):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def subsection(title: str):
    print(f"\n  ── {title} ──")


# ── Layer 1+2+3: Data preparation ─────────────────────────────────────────

def run_data_pipeline(n_per_class: int = 200, add_noise: bool = True) -> tuple:
    """
    Layers 1–3: generate signals → DSP check → extract features.

    Returns:
        X, y, df  (raw signals, labels, feature DataFrame)
    """
    from src.simulation.signal_generator import generate_dataset
    from src.simulation.noise_utils import add_combined_noise
    from src.dsp.fft_analyzer import fft_summary
    from src.features.feature_pipeline import build_feature_dataframe, save_features

    section("LAYER 1 — Signal simulation")
    with Timer("Dataset generation"):
        X, y = generate_dataset(
            n_per_class=n_per_class,
            add_noise=add_noise,
            noise_std=0.05
        )
    print(f"  Signals : {X.shape[0]} total  ({n_per_class} per class × {len(FAULT_TYPES)} classes)")
    print(f"  Shape   : {X.shape}  dtype={X.dtype}")

    # Quick DSP sanity check on one signal per class
    subsection("DSP sanity check (one signal per class)")
    for fault_type, fault_name in FAULT_TYPES.items():
        idx    = (y == fault_type).argmax()
        result = fft_summary(X[idx])
        print(f"  {fault_name:<22} → "
              f"freq={result['dominant_freq']:6.2f} Hz  "
              f"deviation={result['deviation_hz']:+.3f} Hz  "
              f"fundamental_amp={result['fundamental_amplitude']:.4f}")

    section("LAYER 3 — Feature extraction")
    with Timer("Feature extraction"):
        df = build_feature_dataframe(X, y)

    save_features(df)
    print(f"  Shape   : {df.shape}")
    print(f"  Features: {list(df.columns[:6])} ...")
    print(f"  Labels  : {dict(df['fault_name'].value_counts())}")

    return X, y, df


# ── Layer 4: Model training ───────────────────────────────────────────────

def run_training(X, y, df, train_lstm: bool = True) -> dict:
    """
    Layer 4: train Random Forest, Isolation Forest, and optionally LSTM.

    Returns:
        dict with results for each model
    """
    from src.ml.train_classifier import train_classifier
    from src.ml.anomaly_detector import train_anomaly_detector

    results = {}

    section("LAYER 4 — ML model training")

    # Random Forest
    subsection("Random Forest classifier")
    with Timer("Random Forest training"):
        clf_result = train_classifier(df, model_type="random_forest", save=True)
    results["random_forest"] = clf_result
    print(f"  Accuracy  : {clf_result['metrics']['accuracy']:.4f}")
    print(f"  F1 macro  : {clf_result['metrics']['f1_macro']:.4f}")
    print(f"  CV F1     : {clf_result['cv_f1_mean']:.4f} ± {clf_result['cv_f1_std']:.4f}")

    if clf_result["feature_importance"] is not None:
        top5 = clf_result["feature_importance"].head(5)
        print("  Top 5 features:")
        for _, row in top5.iterrows():
            print(f"    {row['feature']:<28} {row['importance']:.4f}")

    # Isolation Forest
    subsection("Isolation Forest anomaly detector")
    with Timer("Isolation Forest training"):
        ano_result = train_anomaly_detector(df, model_type="isolation_forest", save=True)
    results["isolation_forest"] = ano_result
    print(f"  Anomaly precision : {ano_result['metrics']['anomaly_precision']:.4f}")
    print(f"  Anomaly recall    : {ano_result['metrics']['anomaly_recall']:.4f}")
    print(f"  Anomaly F1        : {ano_result['metrics']['anomaly_f1']:.4f}")

    # LSTM
    if train_lstm:
        subsection("LSTM classifier")
        from src.ml.lstm_predictor import train_lstm_classifier
        with Timer("LSTM training"):
            lstm_result = train_lstm_classifier(X, y, save=True)
        results["lstm"] = lstm_result
        print(f"  Accuracy : {lstm_result['metrics']['accuracy']:.4f}")
        print(f"  F1 macro : {lstm_result['metrics']['f1_macro']:.4f}")

    return results


# ── Layer 5: Single signal diagnosis ─────────────────────────────────────

def run_diagnosis(fault_type: int, use_genai: bool = True) -> dict:
    """
    Layers 1–5 on a single signal: generate → DSP → features → ML → GenAI.

    Args:
        fault_type: integer fault label (0–5)
        use_genai:  run Groq LLM explanation (requires GROQ_API_KEY)

    Returns:
        full results dict
    """
    from src.simulation.signal_generator import generate_signal
    from src.simulation.noise_utils import add_combined_noise
    from src.dsp.fft_analyzer import compute_fft, fft_summary
    from src.dsp.spectrogram import detect_transients, detect_voltage_event
    from src.features.feature_pipeline import extract_features, extract_features_vector
    from src.features.thd_calculator import thd_summary
    from src.ml.train_classifier import predict_fault
    from src.ml.anomaly_detector import detect_anomaly

    fault_name = FAULT_TYPES[fault_type]
    section(f"DIAGNOSIS — {fault_name.upper()}")

    # ── Layer 1: Signal ───────────────────────────────────────────────────
    subsection("Signal generation")
    signal = generate_signal(fault_type=fault_type)
    signal = add_combined_noise(signal, snr_db=30)
    print(f"  Samples : {len(signal)}")
    print(f"  Min/Max : {signal.min():.4f} / {signal.max():.4f}")

    # ── Layer 2: DSP ──────────────────────────────────────────────────────
    subsection("DSP analysis")
    fft_data   = fft_summary(signal)
    transients = detect_transients(signal)
    volt_event = detect_voltage_event(signal)

    print(f"  Dominant freq     : {fft_data['dominant_freq']:.3f} Hz")
    print(f"  Frequency dev     : {fft_data['deviation_hz']:+.3f} Hz")
    print(f"  Fundamental amp   : {fft_data['fundamental_amplitude']:.4f}")
    print(f"  Freq deviated     : {fft_data['is_deviated']}")
    print(f"  Transients found  : {transients['n_transients']}")
    print(f"  Voltage event     : {volt_event['event_type']}  "
          f"(amp range {volt_event['min_amplitude']:.3f}–{volt_event['max_amplitude']:.3f})")

    # ── Layer 3: Features ─────────────────────────────────────────────────
    subsection("Feature extraction")
    features    = extract_features(signal)
    feat_vector = extract_features_vector(signal)
    thd_info    = thd_summary(signal)

    print(f"  RMS               : {features['rms']:.4f}")
    print(f"  Peak              : {features['peak_value']:.4f}")
    print(f"  Crest factor      : {features['crest_factor']:.4f}  (nominal 1.414)")
    print(f"  THD               : {thd_info['thd_percent']:.2f}%  (IEEE 519 limit: 5%)")
    print(f"  Dominant harmonic : H{thd_info['dominant_harmonic']}")
    print(f"  Spectral entropy  : {features['spectral_entropy']:.4f}")
    print(f"  Kurtosis          : {features['kurtosis']:.4f}  (normal ≈ 3.0)")
    print(f"  Waveform dev      : {features['waveform_deviation']:.4f}")
    print(f"  Zero crossing rate: {features['zero_crossing_rate']:.1f} crossings/s")

    harmonic_orders = [3, 5, 7, 9, 11]
    print(f"  Harmonic profile  : "
          + "  ".join(f"H{h}={features.get(f'ihd_{h}', 0):.2f}%"
                      for h in harmonic_orders))

    # ── Layer 4: ML ───────────────────────────────────────────────────────
    subsection("ML inference")
    try:
        clf_result = predict_fault(feat_vector)
        print(f"  Classifier  → {clf_result['fault_name']:<22} "
              f"confidence={clf_result['confidence']:.2%}")
        if clf_result["class_probabilities"]:
            for fname, prob in sorted(clf_result["class_probabilities"].items(),
                                      key=lambda x: -x[1])[:3]:
                print(f"    {fname:<26} {prob:.4f}")
    except FileNotFoundError:
        print("  Classifier model not found — run without --skip-training first.")
        clf_result = {
            "fault_name": fault_name,
            "confidence": 1.0,
            "class_probabilities": {},
        }

    try:
        ano_result = detect_anomaly(feat_vector)
        status     = "ANOMALY ⚠" if ano_result["is_anomaly"] else "normal ✓"
        print(f"  Anomaly det → {status:<22} "
              f"score={ano_result['anomaly_score']:.4f}")
    except FileNotFoundError:
        print("  Anomaly model not found — run without --skip-training first.")
        ano_result = {"is_anomaly": False, "anomaly_score": None}

    # ── Layer 5: GenAI ────────────────────────────────────────────────────
    diagnosis = {}
    actions   = []
    report    = ""

    if use_genai:
        subsection("GenAI diagnosis (Groq)")
        try:
            from src.genai.explainer import run_full_diagnosis
            from src.genai.recommender import get_corrective_actions, build_diagnostic_report

            with Timer("Groq LLM calls"):
                diagnosis = run_full_diagnosis(
                    fault_name=clf_result["fault_name"],
                    confidence=clf_result["confidence"],
                    features=features,
                    fft_data=fft_data,
                    anomaly_score=ano_result.get("anomaly_score"),
                    is_anomaly=ano_result.get("is_anomaly", False),
                )
                actions = get_corrective_actions(
                    fault_name=diagnosis["fault_name"],
                    severity=diagnosis["severity"],
                    features=features,
                    fault_explanation=diagnosis["explanation"],
                    use_llm=True,
                )
                report = build_diagnostic_report(
                    fault_name=diagnosis["fault_name"],
                    severity=diagnosis["severity"],
                    confidence=clf_result["confidence"],
                    explanation=diagnosis["explanation"],
                    actions=actions,
                    features=features,
                    fft_data=fft_data,
                )

            print(f"\n  Severity    : {diagnosis['severity']}")
            print(f"  Reason      : {diagnosis.get('reason', '')}")
            print(f"\n  Explanation :\n")
            for line in diagnosis["explanation"].split(". "):
                if line.strip():
                    print(f"    {line.strip()}.")
            print(f"\n  Corrective actions:")
            for act in actions:
                print(f"    [{act['priority']:<12}] {act['action']}")

        except Exception as e:
            print(f"  GenAI error: {e}")
            if "GROQ_API_KEY" in str(e) or "api_key" in str(e).lower():
                print("  → Add GROQ_API_KEY to your .env file.")
            else:
                traceback.print_exc()

    return {
        "signal":       signal,
        "fft_data":     fft_data,
        "features":     features,
        "thd_info":     thd_info,
        "transients":   transients,
        "volt_event":   volt_event,
        "clf_result":   clf_result,
        "ano_result":   ano_result,
        "diagnosis":    diagnosis,
        "actions":      actions,
        "report":       report,
    }


# ── Summary table ─────────────────────────────────────────────────────────

def print_summary(training_results: dict, timing: dict):
    """Print a final summary table of all model metrics and timing."""
    section("PIPELINE SUMMARY")

    print(f"\n  {'Model':<28} {'Metric':<20} {'Value':<10} {'Time':>8}")
    print(f"  {'-'*28} {'-'*20} {'-'*10} {'-'*8}")

    if "random_forest" in training_results:
        r = training_results["random_forest"]
        print(f"  {'Random Forest':<28} {'Accuracy':<20} "
              f"{r['metrics']['accuracy']:.4f}     "
              f"{timing.get('random_forest', 0):>6.1f}s")
        print(f"  {'Random Forest':<28} {'F1 macro':<20} "
              f"{r['metrics']['f1_macro']:.4f}")
        print(f"  {'Random Forest':<28} {'CV F1 (5-fold)':<20} "
              f"{r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.4f}")

    if "isolation_forest" in training_results:
        r = training_results["isolation_forest"]
        print(f"  {'Isolation Forest':<28} {'Anomaly F1':<20} "
              f"{r['metrics']['anomaly_f1']:.4f}     "
              f"{timing.get('isolation_forest', 0):>6.1f}s")
        print(f"  {'Isolation Forest':<28} {'Anomaly recall':<20} "
              f"{r['metrics']['anomaly_recall']:.4f}")

    if "lstm" in training_results:
        r = training_results["lstm"]
        print(f"  {'LSTM classifier':<28} {'Accuracy':<20} "
              f"{r['metrics']['accuracy']:.4f}     "
              f"{timing.get('lstm', 0):>6.1f}s")
        print(f"  {'LSTM classifier':<28} {'F1 macro':<20} "
              f"{r['metrics']['f1_macro']:.4f}")

    total_time = sum(timing.values())
    print(f"\n  Total pipeline time: {total_time:.1f}s")
    print(f"\n  Models saved to : data/models/")
    print(f"  Features saved  : data/processed/features.csv")
    print(f"\n  Run the dashboard:")
    print(f"    streamlit run src/dashboard/app.py")


# ── Main entry point ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Smart Grid Signal Intelligence — end-to-end pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--fault", type=str, default="harmonic_distortion",
        choices=list(FAULT_TYPES.values()),
        help="Fault type to diagnose after training (default: harmonic_distortion)"
    )
    parser.add_argument(
        "--all-faults", action="store_true",
        help="Run diagnosis on all 6 fault types"
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip model training — use existing saved models"
    )
    parser.add_argument(
        "--no-lstm", action="store_true",
        help="Skip LSTM training (faster, use for quick tests)"
    )
    parser.add_argument(
        "--no-genai", action="store_true",
        help="Skip GenAI diagnosis (no Groq API calls)"
    )
    parser.add_argument(
        "--n-per-class", type=int, default=200,
        help="Number of training samples per fault class (default: 200)"
    )
    args = parser.parse_args()

    total_start    = time.time()
    training_results = {}
    timing         = {}

    # ── Training phase ────────────────────────────────────────────────────
    if not args.skip_training:
        t0 = time.time()
        X, y, df = run_data_pipeline(n_per_class=args.n_per_class)
        timing["data"] = time.time() - t0

        t0 = time.time()
        training_results = run_training(X, y, df, train_lstm=not args.no_lstm)
        timing["random_forest"]    = 0
        timing["isolation_forest"] = 0
        if "lstm" in training_results:
            timing["lstm"] = 0

    # ── Diagnosis phase ───────────────────────────────────────────────────
    use_genai = not args.no_genai

    if args.all_faults:
        section("DIAGNOSING ALL FAULT TYPES")
        for fault_type, fault_name in FAULT_TYPES.items():
            try:
                run_diagnosis(fault_type=fault_type, use_genai=use_genai)
            except Exception as e:
                print(f"\n  ERROR diagnosing {fault_name}: {e}")
    else:
        fault_type = [k for k, v in FAULT_TYPES.items() if v == args.fault][0]
        run_diagnosis(fault_type=fault_type, use_genai=use_genai)

    # ── Summary ───────────────────────────────────────────────────────────
    if training_results:
        timing["total"] = time.time() - total_start
        print_summary(training_results, timing)
    else:
        total = time.time() - total_start
        print(f"\n  Pipeline complete in {total:.1f}s")


if __name__ == "__main__":
    main()
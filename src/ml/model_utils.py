"""
Layer 4 — ML
model_utils.py

Shared utilities for model persistence, evaluation metrics,
label encoding, and train/test splitting.
Used by all three ML modules.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)
from config import TEST_SIZE, RANDOM_STATE, MODELS_DIR, FAULT_TYPES
from src.features.feature_pipeline import FEATURE_NAMES


# ── Data preparation ──────────────────────────────────────────────────────

def prepare_features(
    df: pd.DataFrame,
    feature_cols: list = None,
    label_col: str = "label",
    scale: bool = True
) -> tuple:
    """
    Split DataFrame into train/test sets and optionally scale features.

    Args:
        df:           feature DataFrame from feature_pipeline
        feature_cols: list of feature column names (default: FEATURE_NAMES)
        label_col:    name of the label column
        scale:        apply StandardScaler to features

    Returns:
        X_train, X_test, y_train, y_test, scaler (or None if scale=False)
    """
    if feature_cols is None:
        feature_cols = FEATURE_NAMES

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    scaler = None
    if scale:
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def prepare_features_for_lstm(
    X: np.ndarray,
    y: np.ndarray,
    sequence_len: int,
    scale: bool = True
) -> tuple:
    """
    Reshape raw signal array into (samples, timesteps, features=1)
    for LSTM input. Truncates/pads each signal to sequence_len.

    Args:
        X:            raw signal array (n_samples, n_signal_points)
        y:            label array (n_samples,)
        sequence_len: number of time steps per sample
        scale:        normalise each sample to [-1, 1]

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Truncate or pad to sequence_len
    X_seq = X[:, :sequence_len]

    if scale:
        # Per-sample normalisation
        row_max = np.max(np.abs(X_seq), axis=1, keepdims=True) + 1e-12
        X_seq   = X_seq / row_max

    # Add feature dimension: (samples, timesteps, 1)
    X_seq = X_seq[:, :, np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list = None
) -> dict:
    """
    Full evaluation report for a classification model.

    Returns:
        dict with accuracy, f1_macro, confusion_matrix,
              classification_report string, per_class_f1
    """
    if class_names is None:
        class_names = [FAULT_TYPES[i] for i in sorted(FAULT_TYPES.keys())]

    y_pred = model.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="macro")
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)

    per_class_f1 = f1_score(y_test, y_pred, average=None)
    per_class_dict = {
        class_names[i]: round(float(per_class_f1[i]), 4)
        for i in range(len(class_names))
    }

    print(f"\nAccuracy : {acc:.4f}")
    print(f"F1 Macro : {f1:.4f}")
    print(f"\n{report}")

    return {
        "accuracy":               round(float(acc), 4),
        "f1_macro":               round(float(f1), 4),
        "confusion_matrix":       cm.tolist(),
        "classification_report":  report,
        "per_class_f1":           per_class_dict,
    }


def evaluate_anomaly_detector(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anomaly_label: int = -1
) -> dict:
    """
    Evaluate anomaly detector (Isolation Forest style: 1=normal, -1=anomaly).

    Args:
        y_true:        true labels (0=normal, >0=fault)
        y_pred:        predicted labels (1=normal, -1=anomaly)
        anomaly_label: label used for anomalies by the model

    Returns:
        dict with precision, recall, f1 for anomaly class
    """
    # Convert: fault (>0) = anomaly, normal (0) = inlier
    y_true_bin = (y_true > 0).astype(int)       # 1 = fault, 0 = normal
    y_pred_bin = (y_pred == anomaly_label).astype(int)  # 1 = anomaly flag

    report = classification_report(
        y_true_bin, y_pred_bin,
        target_names=["normal", "anomaly"],
        output_dict=True
    )

    print(classification_report(y_true_bin, y_pred_bin,
                                target_names=["normal", "anomaly"]))

    return {
        "anomaly_precision": round(report["anomaly"]["precision"], 4),
        "anomaly_recall":    round(report["anomaly"]["recall"],    4),
        "anomaly_f1":        round(report["anomaly"]["f1-score"],  4),
        "accuracy":          round(report["accuracy"], 4),
    }


# ── Model persistence ─────────────────────────────────────────────────────

def save_model(model, name: str) -> Path:
    """
    Save a sklearn model (or any picklable object) to MODELS_DIR.

    Args:
        model: trained model object
        name:  filename without extension (e.g. 'random_forest')

    Returns:
        Path to saved file
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved → {path}")
    return path


def load_model(name: str):
    """
    Load a pickled model from MODELS_DIR.

    Args:
        name: filename without extension

    Returns:
        loaded model object
    """
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_scaler(scaler: StandardScaler, name: str = "scaler") -> Path:
    """Save a fitted StandardScaler."""
    return save_model(scaler, name)


def load_scaler(name: str = "scaler") -> StandardScaler:
    """Load a fitted StandardScaler."""
    return load_model(name)


def get_feature_importance(model, feature_names: list = None) -> pd.DataFrame:
    """
    Extract feature importances from a tree-based model (Random Forest).

    Args:
        model:         trained sklearn model with feature_importances_
        feature_names: list of feature names

    Returns:
        DataFrame sorted by importance descending
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Model does not have feature_importances_ attribute.")

    importances = model.feature_importances_
    df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return df
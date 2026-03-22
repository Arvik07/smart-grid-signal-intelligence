"""
Layer 4 — ML
anomaly_detector.py

Unsupervised anomaly detection using Isolation Forest.
Trained only on normal signals — detects any deviation without
needing labelled fault data. Complements the supervised classifier.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from config import RANDOM_STATE, FAULT_TYPES
from src.features.feature_pipeline import FEATURE_NAMES
from src.ml.model_utils import (
    prepare_features, evaluate_anomaly_detector,
    save_model, load_model, save_scaler, load_scaler
)


# ── Model definitions ─────────────────────────────────────────────────────

def build_isolation_forest(
    n_estimators: int = 200,
    contamination: float = 0.1,
    random_state: int = RANDOM_STATE
) -> IsolationForest:
    """
    Build an Isolation Forest anomaly detector.

    Args:
        n_estimators:  number of trees
        contamination: expected proportion of anomalies in training data
                       0.1 = assume 10% anomalies during training
        random_state:  reproducibility seed

    Returns:
        Untrained IsolationForest
    """
    return IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )


def build_one_class_svm(
    nu: float = 0.1,
    kernel: str = "rbf",
    gamma: str = "scale"
) -> OneClassSVM:
    """
    Build a One-Class SVM anomaly detector.
    More accurate on small datasets but slower than Isolation Forest.

    Args:
        nu:     upper bound on fraction of margin errors (≈ contamination)
        kernel: 'rbf' | 'linear' | 'poly'
        gamma:  kernel coefficient

    Returns:
        Untrained OneClassSVM
    """
    return OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)


# ── Training pipeline ─────────────────────────────────────────────────────

def train_anomaly_detector(
    df: pd.DataFrame,
    model_type: str = "isolation_forest",
    feature_cols: list = None,
    train_on_normal_only: bool = True,
    save: bool = True
) -> dict:
    """
    Train an anomaly detector on signal features.

    Args:
        df:                   feature DataFrame
        model_type:           'isolation_forest' | 'one_class_svm'
        feature_cols:         feature columns to use
        train_on_normal_only: if True, trains only on class 0 (normal)
                              for a purer anomaly baseline
        save:                 whether to save model and scaler

    Returns:
        dict with keys: model, scaler, metrics
    """
    print(f"\n=== Training {model_type} anomaly detector ===")

    if feature_cols is None:
        feature_cols = FEATURE_NAMES

    # Scale using ALL data's statistics (realistic — scaler sees full distribution)
    _, _, _, _, scaler = prepare_features(df, feature_cols=feature_cols, scale=True)

    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df["label"].values.astype(np.int32)

    X_scaled = scaler.transform(X_all)

    # Train on normal only — purer anomaly baseline
    if train_on_normal_only:
        normal_mask = y_all == 0
        X_train = X_scaled[normal_mask]
        print(f"Training on {X_train.shape[0]} normal samples only")
    else:
        X_train = X_scaled
        print(f"Training on {X_train.shape[0]} samples (all classes)")

    # Build and fit
    if model_type == "isolation_forest":
        model = build_isolation_forest()
    elif model_type == "one_class_svm":
        model = build_one_class_svm()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train)

    # Evaluate on full dataset (test how well it catches faults)
    y_pred   = model.predict(X_scaled)   # 1=normal, -1=anomaly
    metrics  = evaluate_anomaly_detector(y_all, y_pred)

    # Anomaly scores (lower = more anomalous for Isolation Forest)
    if hasattr(model, "score_samples"):
        scores = model.score_samples(X_scaled)
    else:
        scores = -model.decision_function(X_scaled)

    print(f"\nAnomaly score stats:")
    print(f"  Normal  signals: mean={scores[y_all==0].mean():.4f}")
    print(f"  Faulty  signals: mean={scores[y_all>0].mean():.4f}")

    if save:
        save_model(model, model_type)
        save_scaler(scaler, f"scaler_{model_type}")

    return {
        "model":   model,
        "scaler":  scaler,
        "metrics": metrics,
        "scores":  scores,
    }


def detect_anomaly(
    features: np.ndarray,
    model=None,
    scaler=None,
    model_name: str = "isolation_forest"
) -> dict:
    """
    Run anomaly detection on a single feature vector.

    Args:
        features:   1D feature array of shape (n_features,)
        model:      trained anomaly detector (loads from disk if None)
        scaler:     fitted StandardScaler (loads from disk if None)
        model_name: name used when saving

    Returns:
        dict with keys: is_anomaly, anomaly_score, decision
    """
    if model is None:
        model = load_model(model_name)
    if scaler is None:
        scaler = load_scaler(f"scaler_{model_name}")

    features_2d = features.reshape(1, -1)
    if scaler is not None:
        features_2d = scaler.transform(features_2d)

    prediction = int(model.predict(features_2d)[0])  # 1=normal, -1=anomaly
    is_anomaly = prediction == -1

    anomaly_score = None
    if hasattr(model, "score_samples"):
        anomaly_score = float(model.score_samples(features_2d)[0])

    return {
        "is_anomaly":    is_anomaly,
        "anomaly_score": anomaly_score,
        "decision":      "anomaly" if is_anomaly else "normal",
    }


def get_anomaly_threshold(
    model,
    scaler,
    df: pd.DataFrame,
    feature_cols: list = None,
    percentile: float = 5.0
) -> float:
    """
    Compute the anomaly score threshold at a given percentile of
    normal signal scores. Useful for calibrating sensitivity.

    Args:
        model:       trained IsolationForest
        scaler:      fitted StandardScaler
        df:          feature DataFrame
        feature_cols: feature columns
        percentile:  lower percentile of normal scores to use as threshold

    Returns:
        threshold score value
    """
    if feature_cols is None:
        feature_cols = FEATURE_NAMES

    normal_df = df[df["label"] == 0]
    X_normal  = scaler.transform(normal_df[feature_cols].values.astype(np.float32))
    scores    = model.score_samples(X_normal)
    threshold = float(np.percentile(scores, percentile))

    print(f"Anomaly threshold (p{percentile}): {threshold:.4f}")
    return threshold
"""
Layer 4 — ML
train_classifier.py

Random Forest classifier for fault type classification.
Trained on the 25-feature vectors from the feature pipeline.
Classifies signals into 6 fault types: normal, harmonic_distortion,
voltage_sag, voltage_swell, transient_spike, frequency_deviation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from config import RANDOM_STATE, FAULT_TYPES
from src.features.feature_pipeline import FEATURE_NAMES
from src.ml.model_utils import (
    prepare_features, evaluate_classifier,
    save_model, load_model, save_scaler, load_scaler,
    get_feature_importance
)


# ── Model definitions ─────────────────────────────────────────────────────

def build_random_forest(
    n_estimators: int = 200,
    max_depth: int = None,
    min_samples_split: int = 2,
    random_state: int = RANDOM_STATE
) -> RandomForestClassifier:
    """
    Build a Random Forest classifier with sensible defaults.

    Args:
        n_estimators:      number of trees
        max_depth:         max tree depth (None = unlimited)
        min_samples_split: min samples to split a node
        random_state:      reproducibility seed

    Returns:
        Untrained RandomForestClassifier
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=1,
        max_features="sqrt",        # sqrt(n_features) — best for classification
        bootstrap=True,
        oob_score=True,             # out-of-bag score as free validation estimate
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )


def build_gradient_boosting(
    n_estimators: int = 150,
    learning_rate: float = 0.1,
    max_depth: int = 4,
    random_state: int = RANDOM_STATE
) -> GradientBoostingClassifier:
    """
    Build a Gradient Boosting classifier — slower but often more accurate.

    Returns:
        Untrained GradientBoostingClassifier
    """
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )


# ── Training pipeline ─────────────────────────────────────────────────────

def train_classifier(
    df: pd.DataFrame,
    model_type: str = "random_forest",
    feature_cols: list = None,
    save: bool = True
) -> dict:
    """
    Full training pipeline: prepare data → train → evaluate → save.

    Args:
        df:           feature DataFrame from feature_pipeline
        model_type:   'random_forest' | 'gradient_boosting'
        feature_cols: feature columns to use (default: all FEATURE_NAMES)
        save:         whether to save trained model and scaler

    Returns:
        dict with keys: model, scaler, metrics, feature_importance
    """
    print(f"\n=== Training {model_type} classifier ===")

    # 1. Prepare
    X_train, X_test, y_train, y_test, scaler = prepare_features(
        df, feature_cols=feature_cols, scale=True
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # 2. Build model
    if model_type == "random_forest":
        model = build_random_forest()
    elif model_type == "gradient_boosting":
        model = build_gradient_boosting()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 3. Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
    print(f"CV F1 (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 4. Final fit on full train set
    model.fit(X_train, y_train)

    # OOB score — free internal validation estimate (Random Forest only)
    if hasattr(model, "oob_score_"):
        print(f"OOB score (free validation): {model.oob_score_:.4f}")

    # 5. Evaluate
    metrics = evaluate_classifier(model, X_test, y_test)

    # 6. Feature importance (Random Forest only)
    feat_imp = None
    if hasattr(model, "feature_importances_"):
        cols = feature_cols if feature_cols else FEATURE_NAMES
        feat_imp = get_feature_importance(model, cols)
        print("\nTop 10 features:")
        print(feat_imp.head(10).to_string(index=False))

    # 7. Save
    if save:
        save_model(model, model_type)
        if scaler:
            save_scaler(scaler, f"scaler_{model_type}")

    return {
        "model":              model,
        "scaler":             scaler,
        "metrics":            metrics,
        "feature_importance": feat_imp,
        "cv_f1_mean":         round(float(cv_scores.mean()), 4),
        "cv_f1_std":          round(float(cv_scores.std()),  4),
    }


def predict_fault(
    features: np.ndarray,
    model=None,
    scaler=None,
    model_name: str = "random_forest"
) -> dict:
    """
    Predict fault type for a single feature vector.

    Args:
        features:   1D feature array of shape (n_features,)
        model:      trained classifier (loads from disk if None)
        scaler:     fitted StandardScaler (loads from disk if None)
        model_name: name used when saving (for auto-load)

    Returns:
        dict with keys: predicted_label, fault_name,
                        confidence, class_probabilities
    """
    if model is None:
        model = load_model(model_name)
    if scaler is None:
        scaler = load_scaler(f"scaler_{model_name}")

    features_2d = features.reshape(1, -1)
    if scaler is not None:
        features_2d = scaler.transform(features_2d)

    predicted_label = int(model.predict(features_2d)[0])
    fault_name      = FAULT_TYPES.get(predicted_label, "unknown")

    proba = None
    confidence = None
    class_probs = {}

    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(features_2d)[0]
        confidence = float(np.max(proba))
        class_probs = {
            FAULT_TYPES[i]: round(float(p), 4)
            for i, p in enumerate(proba)
            if i in FAULT_TYPES
        }

    return {
        "predicted_label":    predicted_label,
        "fault_name":         fault_name,
        "confidence":         confidence,
        "class_probabilities": class_probs,
    }


def tune_hyperparameters(
    df: pd.DataFrame,
    feature_cols: list = None,
    cv: int = 3
) -> dict:
    """
    Grid search hyperparameter tuning for Random Forest.

    Args:
        df:           feature DataFrame
        feature_cols: feature columns to use
        cv:           cross-validation folds

    Returns:
        dict with best_params and best_score
    """
    print("\n=== Hyperparameter tuning (Random Forest) ===")

    X_train, _, y_train, _, scaler = prepare_features(df, feature_cols=feature_cols)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth":    [None, 10, 20],
        "min_samples_split": [2, 5],
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    print(f"Best params: {grid.best_params_}")
    print(f"Best CV F1:  {grid.best_score_:.4f}")

    return {
        "best_params": grid.best_params_,
        "best_score":  round(grid.best_score_, 4),
        "best_model":  grid.best_estimator_,
    }
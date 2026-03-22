"""
Layer 4 — ML
lstm_predictor.py

LSTM-based time-series model for fault prediction.
Takes raw signal windows as input (not feature vectors) and
predicts fault type directly from the waveform sequence.
Also supports multi-step forecasting to predict future instability.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from config import (
    RANDOM_STATE, FAULT_TYPES, MODELS_DIR,
    LSTM_SEQUENCE_LEN, LSTM_EPOCHS, LSTM_BATCH_SIZE,
    NUM_SAMPLES, SAMPLING_RATE
)
from src.ml.model_utils import prepare_features_for_lstm, evaluate_classifier


def build_lstm_classifier(
    sequence_len: int = LSTM_SEQUENCE_LEN,
    n_classes: int = 6,
    units: int = 64,
    dropout: float = 0.3,
    learning_rate: float = 0.001
):
    """
    Build an LSTM classifier for fault type prediction from raw signals.

    Architecture:
        Input (sequence_len, 1)
        → LSTM(units, return_sequences=True)
        → Dropout
        → LSTM(units // 2)
        → Dropout
        → Dense(32, relu)
        → Dense(n_classes, softmax)

    Args:
        sequence_len:   number of time steps
        n_classes:      number of output classes
        units:          LSTM hidden units
        dropout:        dropout rate
        learning_rate:  Adam optimizer learning rate

    Returns:
        compiled Keras model
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        LSTM(units, return_sequences=True,
             input_shape=(sequence_len, 1)),
        BatchNormalization(),
        Dropout(dropout),

        LSTM(units // 2, return_sequences=False),
        BatchNormalization(),
        Dropout(dropout),

        Dense(32, activation="relu"),
        Dropout(dropout / 2),

        Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_lstm_autoencoder(
    sequence_len: int = LSTM_SEQUENCE_LEN,
    encoding_dim: int = 16,
    units: int = 64
):
    """
    Build an LSTM Autoencoder for unsupervised anomaly detection.
    Trained only on normal signals — high reconstruction error = fault.

    Architecture:
        Encoder: LSTM(units) → RepeatVector → Decoder: LSTM(units) → TimeDistributed(Dense)

    Args:
        sequence_len:  input/output sequence length
        encoding_dim:  bottleneck representation size
        units:         LSTM hidden units

    Returns:
        compiled Keras autoencoder model
    """
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, LSTM, Dense, RepeatVector,
        TimeDistributed, Dropout
    )
    from tensorflow.keras.optimizers import Adam

    inputs = Input(shape=(sequence_len, 1))

    # Encoder
    x = LSTM(units, return_sequences=False)(inputs)
    x = Dropout(0.2)(x)
    encoded = Dense(encoding_dim, activation="relu")(x)

    # Decoder
    x = Dense(units, activation="relu")(encoded)
    x = RepeatVector(sequence_len)(x)
    x = LSTM(units, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    outputs = TimeDistributed(Dense(1))(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(0.001), loss="mse")

    return model


# ── Training pipelines ────────────────────────────────────────────────────

def train_lstm_classifier(
    X: np.ndarray,
    y: np.ndarray,
    sequence_len: int = LSTM_SEQUENCE_LEN,
    epochs: int = LSTM_EPOCHS,
    batch_size: int = LSTM_BATCH_SIZE,
    save: bool = True
) -> dict:
    """
    Full LSTM classifier training pipeline.

    Args:
        X:            raw signal array (n_samples, n_signal_points)
        y:            label array (n_samples,)
        sequence_len: time steps per sample
        epochs:       training epochs
        batch_size:   mini-batch size
        save:         save model weights to MODELS_DIR

    Returns:
        dict with model, history, metrics
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )

    print(f"\n=== Training LSTM classifier ===")

    # 1. Prepare sequences
    X_train, X_test, y_train, y_test = prepare_features_for_lstm(
        X, y, sequence_len=sequence_len, scale=True
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # 2. Build model
    n_classes = len(np.unique(y))
    model = build_lstm_classifier(sequence_len=sequence_len, n_classes=n_classes)
    model.summary()

    # 3. Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = str(MODELS_DIR / "lstm_best.h5")
        callbacks.append(
            ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_loss")
        )

    # 4. Train
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # 5. Evaluate
    y_pred   = np.argmax(model.predict(X_test), axis=1)
    metrics  = evaluate_classifier(
        type("M", (), {"predict": lambda s, x: np.argmax(model.predict(x), axis=1)})(),
        X_test, y_test
    )

    # 6. Save final model
    if save:
        final_path = MODELS_DIR / "lstm_classifier.h5"
        model.save(str(final_path))
        print(f"LSTM saved → {final_path}")

    return {
        "model":   model,
        "history": history.history,
        "metrics": metrics,
    }


def train_lstm_autoencoder(
    X: np.ndarray,
    y: np.ndarray,
    sequence_len: int = LSTM_SEQUENCE_LEN,
    epochs: int = 20,
    batch_size: int = LSTM_BATCH_SIZE,
    save: bool = True
) -> dict:
    """
    Train the LSTM autoencoder on normal signals only.
    High reconstruction error at inference → anomaly.

    Returns:
        dict with model, threshold (95th percentile of normal errors)
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    print(f"\n=== Training LSTM autoencoder (anomaly detection) ===")

    # Use only normal signals for training
    X_seq = X[:, :sequence_len, np.newaxis]
    row_max = np.max(np.abs(X_seq), axis=1, keepdims=True) + 1e-12
    X_seq = X_seq / row_max

    X_normal = X_seq[y == 0]
    print(f"Training autoencoder on {len(X_normal)} normal samples")

    model = build_lstm_autoencoder(sequence_len=sequence_len)

    history = model.fit(
        X_normal, X_normal,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )

    # Compute reconstruction errors on normal signals
    recon_normal = model.predict(X_normal)
    errors_normal = np.mean(np.square(X_normal - recon_normal), axis=(1, 2))

    # Set threshold at 95th percentile of normal errors
    threshold = float(np.percentile(errors_normal, 95))
    print(f"Anomaly threshold (95th pct of normal): {threshold:.6f}")

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(str(MODELS_DIR / "lstm_autoencoder.h5"))
        import pickle
        with open(MODELS_DIR / "lstm_ae_threshold.pkl", "wb") as f:
            pickle.dump(threshold, f)

    return {
        "model":     model,
        "threshold": threshold,
        "history":   history.history,
    }


# ── Inference ─────────────────────────────────────────────────────────────

def predict_with_lstm(
    signal: np.ndarray,
    model=None,
    sequence_len: int = LSTM_SEQUENCE_LEN,
    model_path: str = None
) -> dict:
    """
    Predict fault type from a raw signal using LSTM classifier.

    Args:
        signal:       raw waveform (n_points,)
        model:        loaded Keras model (loads from file if None)
        sequence_len: time steps to use
        model_path:   path to .h5 file (optional override)

    Returns:
        dict with predicted_label, fault_name, confidence, probabilities
    """
    import tensorflow as tf

    if model is None:
        path = model_path or str(MODELS_DIR / "lstm_classifier.h5")
        model = tf.keras.models.load_model(path)

    # Prepare input
    seq   = signal[:sequence_len]
    s_max = np.max(np.abs(seq)) + 1e-12
    seq   = (seq / s_max).reshape(1, sequence_len, 1)

    proba = model.predict(seq, verbose=0)[0]
    label = int(np.argmax(proba))

    return {
        "predicted_label": label,
        "fault_name":      FAULT_TYPES.get(label, "unknown"),
        "confidence":      round(float(np.max(proba)), 4),
        "probabilities":   {
            FAULT_TYPES[i]: round(float(p), 4)
            for i, p in enumerate(proba)
            if i in FAULT_TYPES
        },
    }


def compute_reconstruction_error(
    signal: np.ndarray,
    model=None,
    sequence_len: int = LSTM_SEQUENCE_LEN,
    threshold: float = None
) -> dict:
    """
    Compute autoencoder reconstruction error for anomaly detection.

    Returns:
        dict with reconstruction_error, is_anomaly, threshold
    """
    import tensorflow as tf
    import pickle

    if model is None:
        model = tf.keras.models.load_model(str(MODELS_DIR / "lstm_autoencoder.h5"))

    if threshold is None:
        thr_path = MODELS_DIR / "lstm_ae_threshold.pkl"
        if thr_path.exists():
            with open(thr_path, "rb") as f:
                threshold = pickle.load(f)
        else:
            threshold = 0.01  # fallback

    seq   = signal[:sequence_len]
    s_max = np.max(np.abs(seq)) + 1e-12
    seq   = (seq / s_max).reshape(1, sequence_len, 1)

    recon = model.predict(seq, verbose=0)
    error = float(np.mean(np.square(seq - recon)))

    return {
        "reconstruction_error": round(error, 6),
        "threshold":            round(threshold, 6),
        "is_anomaly":           error > threshold,
    }
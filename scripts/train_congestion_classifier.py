from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_sample_weight

# ---------------------------------------------------------------------------
# Project import
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BACKEND_ROOT = PROJECT_ROOT / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.classification.congestion_classifier import (  # noqa: E402
    CLASS_NAMES,
    CongestionClassifier,
    CongestionClassifierConfig,
)
from app.ml.models.gru_model import TrafficGRUModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "congestion"
)

SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences"
)

GRU_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "gru_residual_model.pt"
)

RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR = (
    RESULTS_DIR / "models"
)

PREDICTION_DIR = (
    RESULTS_DIR / "predictions"
)

METRICS_DIR = (
    RESULTS_DIR / "metrics"
)

FIGURES_DIR = (
    RESULTS_DIR / "figures"
)

for directory in (
    MODEL_DIR,
    PREDICTION_DIR,
    METRICS_DIR,
    FIGURES_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# This controls only the TRAINING data.
#
# Validation and test are ALWAYS evaluated completely.
MAX_TRAINING_OBSERVATIONS = 1_000_000

RANDOM_STATE = 42

SENSOR_FEATURE_NAME = "sensor_index"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_split(
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load compact classification features and labels."""

    X_path = DATA_DIR / f"X_{split}.npy"
    y_path = DATA_DIR / f"y_{split}.npy"

    if not X_path.exists():
        raise FileNotFoundError(
            f"Missing feature file: {X_path}"
        )

    if not y_path.exists():
        raise FileNotFoundError(
            f"Missing label file: {y_path}"
        )

    X = np.load(
        X_path,
        allow_pickle=False,
    )

    y = np.load(
        y_path,
        allow_pickle=False,
    )

    if X.ndim != 2:
        raise ValueError(
            f"{split}: expected X to be 2-D, got {X.shape}"
        )

    if y.ndim != 2:
        raise ValueError(
            f"{split}: expected y to be 2-D, got {y.shape}"
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{split}: sample mismatch: "
            f"{X.shape[0]} != {y.shape[0]}"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"{split}: X contains invalid values."
        )

    return (
        X.astype(np.float32),
        y.astype(np.int8),
    )


# ---------------------------------------------------------------------------
# Convert multi-sensor target to sample-level observations
# ---------------------------------------------------------------------------


def expand_sensor_observations(
    X: np.ndarray,
    y: np.ndarray,
    gru_predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert timestamp-level features into sensor-level observations.

    Input:
        X:
            (samples, 24 historical features)

        y:
            (samples, 207 congestion labels)

        gru_predictions:
            (samples, 207 predicted future speeds)

    Output:
        X_expanded:
            (samples * 207, 26)

        y_expanded:
            (samples * 207,)

    The 26 classifier features are:

        24 historical/context features
        1 GRU predicted future speed
        1 sensor index
    """

    samples, feature_count = X.shape
    y_samples, sensors = y.shape

    if samples != y_samples:
        raise ValueError(
            "X/y sample count mismatch: "
            f"{samples} != {y_samples}"
        )

    if gru_predictions.shape != (
        samples,
        sensors,
    ):
        raise ValueError(
            "GRU prediction shape mismatch: "
            f"{gru_predictions.shape}; "
            f"expected {(samples, sensors)}"
        )

    if feature_count != 24:
        raise ValueError(
            "Expected 24 historical features, "
            f"got {feature_count}"
        )

    if not np.isfinite(gru_predictions).all():
        raise ValueError(
            "GRU predictions contain "
            "NaN or infinite values."
        )

    repeated_features = np.repeat(
        X,
        sensors,
        axis=0,
    )

    repeated_gru_predictions = (
        gru_predictions.reshape(-1, 1)
    )

    sensor_indices = np.tile(
        np.arange(sensors),
        samples,
    ).astype(
        np.float32
    ).reshape(-1, 1)

    expanded_X = np.column_stack(
        [
            repeated_features,
            repeated_gru_predictions,
            sensor_indices,
        ]
    ).astype(
        np.float32,
        copy=False,
    )

    expanded_y = y.reshape(-1)

    expected_features = 26

    if expanded_X.shape != (
        samples * sensors,
        expected_features,
    ):
        raise RuntimeError(
            "Expanded feature shape mismatch: "
            f"{expanded_X.shape}; expected "
            f"({samples * sensors}, "
            f"{expected_features})"
        )

    return (
        expanded_X,
        expanded_y,
    )


# ---------------------------------------------------------------------------
# Training subsampling
# ---------------------------------------------------------------------------


def limit_training_data(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Limit training observations without modifying validation/test."""

    if len(X) <= MAX_TRAINING_OBSERVATIONS:
        return X, y

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    indices = rng.choice(
        len(X),
        size=MAX_TRAINING_OBSERVATIONS,
        replace=False,
    )

    indices.sort()

    return (
        X[indices],
        y[indices],
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate classification metrics."""

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Save normalized confusion matrix."""

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        normalize="true",
    )

    figure, axis = plt.subplots(
        figsize=(6, 5)
    )

    image = axis.imshow(
        matrix,
    )

    axis.set_xticks(
        [0, 1, 2],
        CLASS_NAMES,
    )

    axis.set_yticks(
        [0, 1, 2],
        CLASS_NAMES,
    )

    axis.set_xlabel(
        "Predicted class"
    )

    axis.set_ylabel(
        "Actual class"
    )

    axis.set_title(
        "Normalized Congestion Classification Confusion Matrix"
    )

    for row in range(3):
        for column in range(3):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
            )

    figure.colorbar(
        image,
        ax=axis,
    )

    figure.tight_layout()

    output = (
        FIGURES_DIR
        / "congestion_confusion_matrix.png"
    )

    figure.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Saved: {output}"
    )


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def save_feature_importance(
    classifier: CongestionClassifier,
    feature_names: list[str],
) -> None:
    """Save and visualize XGBoost feature importance."""

    importance = classifier.get_feature_importance()

    if len(importance) != len(feature_names):
        raise RuntimeError(
            "Feature importance count does not match feature names."
        )

    dataframe = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    csv_path = (
        RESULTS_DIR
        / "classification"
        / "congestion_feature_importance.csv"
    )

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        csv_path,
        index=False,
    )

    top = dataframe.head(
        15
    ).sort_values(
        "importance",
    )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.barh(
        top["feature"],
        top["importance"],
    )

    axis.set_xlabel(
        "Feature importance"
    )

    axis.set_ylabel(
        "Feature"
    )

    axis.set_title(
        "Congestion Classifier Feature Importance"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "congestion_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Saved: {csv_path}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_gru_predictions(
    split: str,
) -> np.ndarray:
    """
    Generate future-speed predictions using the trained GRU.

    The sequence split and classification split must contain
    the same samples in the same order.
    """

    sequence_path = (
        SEQUENCE_DIR
        / f"X_{split}.npy"
    )

    if not sequence_path.exists():
        raise FileNotFoundError(
            f"Missing sequence file: {sequence_path}"
        )

    if not GRU_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing GRU model: {GRU_MODEL_PATH}"
        )

    sequences = np.load(
        sequence_path,
        allow_pickle=False,
    ).astype(
        np.float32
    )

    if sequences.ndim != 3:
        raise ValueError(
            f"{split}: expected GRU input to be 3-D, "
            f"got {sequences.shape}"
        )

    if sequences.shape[1:] != (
        12,
        207,
    ):
        raise ValueError(
            f"{split}: unexpected GRU sequence shape "
            f"{sequences.shape}"
        )

    gru = TrafficGRUModel()

    gru.load_checkpoint(
        GRU_MODEL_PATH,
    )

    predictions = gru.predict(
        sequences,
    )

    if predictions.shape != (
        len(sequences),
        207,
    ):
        raise RuntimeError(
            f"{split}: unexpected GRU prediction shape "
            f"{predictions.shape}"
        )

    if not np.isfinite(predictions).all():
        raise RuntimeError(
            f"{split}: GRU predictions contain "
            "NaN or infinite values."
        )

    print(
        f"{split}: GRU predictions = "
        f"{predictions.shape}"
    )

    return predictions.astype(
        np.float32,
        copy=False,
    )

def main() -> None:
    """Train and evaluate the congestion classifier."""

    print("=" * 64)
    print("Sprint 4.4 — Congestion Classification")
    print("=" * 64)

    # -----------------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------------

    print(
        "\nLoading classification datasets..."
    )

    X_train, y_train = load_split(
        "train"
    )

    X_validation, y_validation = load_split(
        "validation"
    )

    X_test, y_test = load_split(
        "test"
    )

    print(
        "\nGenerating GRU future-speed predictions..."
    )

    gru_train = generate_gru_predictions(
        "train"
    )

    gru_validation = generate_gru_predictions(
        "validation"
    )

    gru_test = generate_gru_predictions(
        "test"
    )

    if len(X_train) != len(gru_train):
        raise RuntimeError(
            "Train classification features and GRU "
            "predictions have different sample counts."
        )

    if len(X_validation) != len(gru_validation):
        raise RuntimeError(
            "Validation classification features and GRU "
            "predictions have different sample counts."
        )

    if len(X_test) != len(gru_test):
        raise RuntimeError(
            "Test classification features and GRU "
            "predictions have different sample counts."
        )

    print(
        f"Train:       X={X_train.shape}, y={y_train.shape}"
    )

    print(
        f"Validation:  X={X_validation.shape}, y={y_validation.shape}"
    )

    print(
        f"Test:        X={X_test.shape}, y={y_test.shape}"
    )

    # -----------------------------------------------------------------------
    # Expand sensors
    # -----------------------------------------------------------------------

    print(
        "\nExpanding sensor observations..."
    )

    X_train, y_train = expand_sensor_observations(
        X_train,
        y_train,
        gru_train,
    )

    X_validation, y_validation = expand_sensor_observations(
        X_validation,
        y_validation,
        gru_validation,
    )

    X_test, y_test = expand_sensor_observations(
        X_test,
        y_test,
        gru_test,
    )

    print(
        f"Expanded train:      {X_train.shape}"
    )

    print(
        f"Expanded validation: {X_validation.shape}"
    )

    print(
        f"Expanded test:       {X_test.shape}"
    )

    # -----------------------------------------------------------------------
    # Limit training only
    # -----------------------------------------------------------------------

    original_train_size = len(
        X_train
    )

    X_train, y_train = limit_training_data(
        X_train,
        y_train,
    )

    print(
        "\nTraining observations"
    )

    print(
        "====================="
    )

    print(
        f"Original: {original_train_size:,}"
    )

    print(
        f"Used:     {len(X_train):,}"
    )

    print(
        f"Validation: {len(X_validation):,}"
    )

    print(
        f"Test:       {len(X_test):,}"
    )

    # -----------------------------------------------------------------------
    # Class weights
    # -----------------------------------------------------------------------

    print(
        "\nCalculating class weights..."
    )

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    ).astype(
        np.float32
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    config = CongestionClassifierConfig(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    classifier = CongestionClassifier(
        config
    )

    print(
        "\nTraining classifier..."
    )

    classifier.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    # -----------------------------------------------------------------------
    # Predictions
    # -----------------------------------------------------------------------

    print(
        "\nGenerating validation predictions..."
    )

    validation_predictions = classifier.predict(
        X_validation
    )

    validation_probabilities = classifier.predict_proba(
        X_validation
    )

    print(
        "Generating test predictions..."
    )

    test_predictions = classifier.predict(
        X_test
    )

    test_probabilities = classifier.predict_proba(
        X_test
    )

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    print(
        "\nCongestion Classification — Validation"
    )

    print(
        "=" * 64
    )

    for name, value in validation_metrics.items():
        print(
            f"{name:<20}: {value:.4f}"
        )

    print(
        "\nCongestion Classification — Test"
    )

    print(
        "=" * 64
    )

    for name, value in test_metrics.items():
        print(
            f"{name:<20}: {value:.4f}"
        )

    # -----------------------------------------------------------------------
    # Per-class report
    # -----------------------------------------------------------------------

    report = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    print(
        "\nTest classification report"
    )

    print(
        "==========================="
    )

    print(
        report
    )

    report_path = (
        METRICS_DIR
        / "congestion_classification_report.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    # -----------------------------------------------------------------------
    # Save predictions
    # -----------------------------------------------------------------------

    np.save(
        PREDICTION_DIR
        / "congestion_validation_predictions.npy",
        validation_predictions,
    )

    np.save(
        PREDICTION_DIR
        / "congestion_test_predictions.npy",
        test_predictions,
    )

    np.save(
        PREDICTION_DIR
        / "congestion_validation_probabilities.npy",
        validation_probabilities,
    )

    np.save(
        PREDICTION_DIR
        / "congestion_test_probabilities.npy",
        test_probabilities,
    )

    # -----------------------------------------------------------------------
    # Save metrics
    # -----------------------------------------------------------------------

    metrics = {
        "model": "shared_xgboost_congestion_classifier",
        "classes": list(CLASS_NAMES),
        "label_definition": {
            "low": "CI < 0.10",
            "moderate": "0.10 <= CI < 0.30",
            "severe": "CI >= 0.30",
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "training_observations_original": (
            int(original_train_size)
        ),
        "training_observations_used": (
            int(len(X_train))
        ),
        "validation_observations": (
            int(len(X_validation))
        ),
        "test_observations": (
            int(len(X_test))
        ),
        "model_configuration": (
            classifier.get_booster_config()
        ),
    }

    metrics_path = (
        METRICS_DIR
        / "congestion_classification_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------------
    # Save model
    # -----------------------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "congestion_classifier.joblib"
    )

    joblib.dump(
        classifier,
        model_path,
    )

    # -----------------------------------------------------------------------
    # Feature names
    # -----------------------------------------------------------------------

    feature_names = [
        "current_speed_mean",
        "current_speed_std",
        "current_speed_min",
        "current_speed_max",
        "previous_speed_mean",
        "previous_speed_std",
        "speed_change_mean",
        "speed_change_std",
        "long_term_speed_change",
        "sequence_speed_mean",
        "sequence_speed_std",
        "recent_speed_mean",
        "recent_speed_std",
        "near_zero_fraction",
        "low_speed_fraction",
        "moderate_speed_fraction",
        "high_speed_fraction",
        "hour",
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "day_sin",
        "day_cos",
        "is_weekend",
        "gru_predicted_speed",
        SENSOR_FEATURE_NAME,
    ]

    save_feature_importance(
        classifier,
        feature_names,
    )

    # -----------------------------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------------------------

    save_confusion_matrix(
        y_test,
        test_predictions,
    )

    # -----------------------------------------------------------------------
    # Save metadata
    # -----------------------------------------------------------------------

    metadata = {
        "model": "shared_xgboost_congestion_classifier",
        "sensor_count": 207,
        "input_feature_count": len(
            feature_names
        ),
        "forecast_dependency": (
            "GRU predicted future speed is used as "
            "an XGBoost classifier feature."
        ),
        "feature_names": feature_names,
        "class_names": list(
            CLASS_NAMES
        ),
        "training_cap": (
            MAX_TRAINING_OBSERVATIONS
        ),
        "class_weighting": "balanced",
        "reference_speed": (
            "sensor-specific training-data P85"
        ),
        "congestion_index": (
            "1 - future_speed / reference_speed"
        ),
        "label_thresholds": {
            "low_moderate": 0.10,
            "moderate_severe": 0.30,
        },
    }

    metadata_path = (
        METRICS_DIR
        / "congestion_classifier_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nArtifacts saved"
    )

    print(
        "==============="
    )

    print(
        f"Model:   {model_path}"
    )

    print(
        f"Metrics: {metrics_path}"
    )

    print(
        f"Report:  {report_path}"
    )

    print(
        "\nSprint 4.4 completed."
    )


if __name__ == "__main__":
    main()
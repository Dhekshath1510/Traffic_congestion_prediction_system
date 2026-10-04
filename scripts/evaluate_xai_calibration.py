"""
Sprint 4.7
XAI Error Analysis and Probability Calibration.

Evaluates the existing shared XGBoost congestion classifier.

Analyses
--------
1. Prediction confidence vs correctness
2. Per-class confidence
3. Expected Calibration Error (ECE)
4. Multiclass Brier score
5. Confusion-pair error analysis
6. Error-specific TreeSHAP analysis
7. Sensor-level classification errors

The serialized project classifier is a wrapper around an XGBoost estimator.
TreeSHAP does not operate on the project wrapper directly, so this module
explicitly extracts the underlying XGBoost estimator before constructing
TreeExplainer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Final

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
BACKEND_DIR: Final[Path] = PROJECT_ROOT / "backend"

# The serialized classifier contains references to the backend `app` package.
# This must be configured before joblib unpickles the model.
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODEL_PATH: Final[Path] = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "congestion_classifier.joblib"
)

X_TEST_PATH: Final[Path] = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "congestion"
    / "X_test.npy"
)

Y_TEST_PATH: Final[Path] = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "congestion"
    / "y_test.npy"
)

METRICS_DIR: Final[Path] = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

EXPLAINABILITY_DIR: Final[Path] = (
    PROJECT_ROOT
    / "results"
    / "explainability"
)

FIGURES_DIR: Final[Path] = (
    PROJECT_ROOT
    / "results"
    / "figures"
)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

RANDOM_SEED: Final[int] = 42
SHAP_SAMPLE_SIZE: Final[int] = 10_000
CALIBRATION_BIN_COUNT: Final[int] = 10

CLASS_NAMES: Final[list[str]] = [
    "LOW",
    "MODERATE",
    "SEVERE",
]

FEATURE_NAMES: Final[list[str]] = [
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
    "sensor_index",
]


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Array = np.ndarray


# ---------------------------------------------------------------------------
# Directory management
# ---------------------------------------------------------------------------

def ensure_directories() -> None:
    """Create all Sprint 4.7 output directories."""
    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXPLAINABILITY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model() -> Any:
    """
    Load the serialized project classifier.

    Returns
    -------
    Any
        The project-level classifier wrapper.

    Notes
    -----
    The model is intentionally loaded as `Any` because the project wrapper
    is serialized through joblib and its concrete type is resolved at runtime.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Classifier model not found: {MODEL_PATH}"
        )

    model: Any = joblib.load(MODEL_PATH)

    return model


def extract_xgboost_classifier(model: Any) -> XGBClassifier:
    """
    Extract the underlying XGBClassifier from the project wrapper.

    The project uses a custom CongestionClassifier wrapper. TreeSHAP does not
    support that wrapper directly, but it supports the underlying XGBClassifier.

    Several conventional estimator attribute names are checked to make this
    function robust against minor wrapper implementation differences.
    """
    if isinstance(model, XGBClassifier):
        return model

    candidate_attribute_names: tuple[str, ...] = (
        "model",
        "classifier",
        "estimator",
        "xgb_model",
        "xgb_classifier",
        "booster",
        "_model",
        "_classifier",
        "_estimator",
    )

    for attribute_name in candidate_attribute_names:
        try:
            candidate: Any = getattr(
                model,
                attribute_name,
                None,
            )
        except Exception:
            continue

        if isinstance(candidate, XGBClassifier):
            return candidate

    model_type = type(model).__name__

    raise TypeError(
        "Unable to extract the underlying XGBClassifier from the "
        f"serialized classifier wrapper ({model_type}). "
        "Expected one of the wrapper attributes: "
        + ", ".join(candidate_attribute_names)
    )


def load_data() -> tuple[Any, Array, Array]:
    """
    Load classifier and test data.
    """
    model = load_model()

    if not X_TEST_PATH.exists():
        raise FileNotFoundError(
            f"X_test not found: {X_TEST_PATH}"
        )

    if not Y_TEST_PATH.exists():
        raise FileNotFoundError(
            f"y_test not found: {Y_TEST_PATH}"
        )

    x_test: Array = np.load(X_TEST_PATH)
    y_test: Array = np.load(Y_TEST_PATH)

    return model, x_test, y_test


# ---------------------------------------------------------------------------
# Feature expansion
# ---------------------------------------------------------------------------

def expand_sensor_features(
    x_test: Array,
    y_test: Array,
) -> tuple[Array, Array]:
    """
    Reproduce the exact 25-feature representation used by the classifier.

    Input
    -----
    x_test:
        Shape (samples, 24)

    y_test:
        Shape (samples, sensors)

    Output
    ------
    expanded_features:
        Shape (samples * sensors, 25)

    expanded_targets:
        Shape (samples * sensors,)
    """
    if x_test.ndim != 2:
        raise ValueError(
            "X_test must be two-dimensional. "
            f"Received shape: {x_test.shape}"
        )

    if y_test.ndim != 2:
        raise ValueError(
            "y_test must be two-dimensional. "
            f"Received shape: {y_test.shape}"
        )

    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            "X_test and y_test must contain the same number of samples. "
            f"X_test={x_test.shape[0]}, "
            f"y_test={y_test.shape[0]}"
        )

    if x_test.shape[1] != 24:
        raise ValueError(
            "Expected 24 engineered features before adding sensor_index. "
            f"Received {x_test.shape[1]}"
        )

    n_samples: int = x_test.shape[0]
    n_sensors: int = y_test.shape[1]

    repeated_features: Array = np.repeat(
        x_test,
        repeats=n_sensors,
        axis=0,
    )

    sensor_indices: Array = np.tile(
        np.arange(
            n_sensors,
            dtype=np.float32,
        ),
        n_samples,
    ).reshape(-1, 1)

    expanded_features: Array = np.hstack(
        (
            repeated_features,
            sensor_indices,
        )
    ).astype(
        np.float32,
        copy=False,
    )

    expanded_targets: Array = y_test.reshape(-1).astype(
        np.int64,
        copy=False,
    )

    if expanded_features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            "Expanded feature count does not match FEATURE_NAMES. "
            f"Features={expanded_features.shape[1]}, "
            f"Names={len(FEATURE_NAMES)}"
        )

    return expanded_features, expanded_targets


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def generate_predictions(
    model: Any,
    x_expanded: Array,
) -> tuple[Array, Array]:
    """
    Generate predictions using the project wrapper.

    The wrapper remains responsible for inference so that evaluation matches
    the original classifier behavior exactly.
    """
    predictions: Array = np.asarray(
        model.predict(x_expanded)
    ).astype(
        np.int64,
        copy=False,
    )

    probabilities: Array = np.asarray(
        model.predict_proba(x_expanded)
    ).astype(
        np.float64,
        copy=False,
    )

    if probabilities.ndim != 2:
        raise ValueError(
            "Classifier probabilities must be two-dimensional. "
            f"Received shape: {probabilities.shape}"
        )

    if probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            "Expected three class probabilities. "
            f"Received shape: {probabilities.shape}"
        )

    if predictions.shape[0] != x_expanded.shape[0]:
        raise ValueError(
            "Prediction count does not match feature count."
        )

    return predictions, probabilities


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def calculate_confidence(
    probabilities: Array,
) -> Array:
    """Return maximum predicted probability for every observation."""
    return probabilities.max(axis=1)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calculate_ece(
    y_true: Array,
    probabilities: Array,
    n_bins: int = CALIBRATION_BIN_COUNT,
) -> tuple[float, pd.DataFrame]:
    """
    Calculate multiclass Expected Calibration Error.

    Confidence is the maximum predicted class probability.
    Accuracy is measured against the corresponding argmax prediction.
    """
    predictions: Array = probabilities.argmax(
        axis=1
    )

    confidence: Array = probabilities.max(
        axis=1
    )

    correctness: Array = (
        predictions == y_true
    ).astype(
        np.float64
    )

    bin_edges: Array = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    rows: list[dict[str, float | int]] = []

    ece: float = 0.0

    total_samples: int = len(y_true)

    for bin_index in range(n_bins):
        lower: float = float(
            bin_edges[bin_index]
        )

        upper: float = float(
            bin_edges[bin_index + 1]
        )

        if bin_index == n_bins - 1:
            mask: Array = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        count: int = int(mask.sum())

        if count == 0:
            continue

        bin_accuracy: float = float(
            correctness[mask].mean()
        )

        bin_confidence: float = float(
            confidence[mask].mean()
        )

        fraction: float = count / total_samples

        calibration_gap: float = abs(
            bin_accuracy - bin_confidence
        )

        ece += fraction * calibration_gap

        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "mean_confidence": bin_confidence,
                "accuracy": bin_accuracy,
                "calibration_gap": calibration_gap,
            }
        )

    return float(ece), pd.DataFrame(rows)


def calculate_multiclass_brier_score(
    y_true: Array,
    probabilities: Array,
    n_classes: int,
) -> float:
    """
    Calculate the multiclass Brier score.

    Uses one-hot target vectors and the squared Euclidean distance between
    the target distribution and predicted probability distribution.
    """
    one_hot: Array = np.eye(
        n_classes,
        dtype=np.float64,
    )[y_true]

    return float(
        np.mean(
            np.sum(
                np.square(
                    probabilities - one_hot
                ),
                axis=1,
            )
        )
    )


# ---------------------------------------------------------------------------
# Confidence/error analysis
# ---------------------------------------------------------------------------

def confidence_analysis(
    y_true: Array,
    predictions: Array,
    probabilities: Array,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate per-class confidence and confusion-pair analyses."""
    confidence: Array = calculate_confidence(
        probabilities
    )

    correct: Array = (
        predictions == y_true
    )

    class_rows: list[dict[str, Any]] = []

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):
        actual_mask: Array = (
            y_true == class_index
        )

        sample_count: int = int(
            actual_mask.sum()
        )

        if sample_count == 0:
            continue

        class_confidence: Array = confidence[
            actual_mask
        ]

        class_correct: Array = correct[
            actual_mask
        ]

        class_rows.append(
            {
                "class": class_name,
                "sample_count": sample_count,
                "mean_confidence": float(
                    class_confidence.mean()
                ),
                "median_confidence": float(
                    np.median(class_confidence)
                ),
                "accuracy": float(
                    class_correct.mean()
                ),
                "high_confidence_fraction": float(
                    (
                        class_confidence >= 0.8
                    ).mean()
                ),
                "low_confidence_fraction": float(
                    (
                        class_confidence < 0.5
                    ).mean()
                ),
            }
        )

    confidence_by_class = pd.DataFrame(
        class_rows
    )

    error_rows: list[dict[str, Any]] = []

    for true_index, true_class in enumerate(
        CLASS_NAMES
    ):
        for predicted_index, predicted_class in enumerate(
            CLASS_NAMES
        ):
            mask: Array = (
                (y_true == true_index)
                & (predictions == predicted_index)
            )

            count: int = int(mask.sum())

            if count == 0:
                continue

            pair_confidence: Array = confidence[
                mask
            ]

            error_rows.append(
                {
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "count": count,
                    "fraction_of_dataset": float(
                        count / len(y_true)
                    ),
                    "mean_confidence": float(
                        pair_confidence.mean()
                    ),
                    "median_confidence": float(
                        np.median(pair_confidence)
                    ),
                    "correct": (
                        true_index == predicted_index
                    ),
                }
            )

    error_analysis = pd.DataFrame(
        error_rows
    )

    return confidence_by_class, error_analysis


# ---------------------------------------------------------------------------
# Sensor analysis
# ---------------------------------------------------------------------------

def calculate_sensor_errors(
    y_test: Array,
    predictions: Array,
) -> pd.DataFrame:
    """Calculate classification errors independently for every sensor."""
    n_samples: int
    n_sensors: int

    n_samples, n_sensors = y_test.shape

    if predictions.size != n_samples * n_sensors:
        raise ValueError(
            "Prediction count cannot be reshaped to the test sensor matrix. "
            f"Predictions={predictions.size}, "
            f"expected={n_samples * n_sensors}"
        )

    prediction_matrix: Array = predictions.reshape(
        n_samples,
        n_sensors,
    )

    rows: list[dict[str, Any]] = []

    for sensor_index in range(n_sensors):
        sensor_true: Array = y_test[
            :,
            sensor_index,
        ]

        sensor_prediction: Array = prediction_matrix[
            :,
            sensor_index,
        ]

        accuracy: float = float(
            np.mean(
                sensor_true == sensor_prediction
            )
        )

        severe_mask: Array = (
            sensor_true == 2
        )

        severe_recall: float | None

        if severe_mask.any():
            severe_recall = float(
                np.mean(
                    sensor_prediction[
                        severe_mask
                    ] == 2
                )
            )
        else:
            severe_recall = None

        rows.append(
            {
                "sensor_index": sensor_index,
                "accuracy": accuracy,
                "error_rate": 1.0 - accuracy,
                "severe_recall": severe_recall,
                "sample_count": n_samples,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TreeSHAP
# ---------------------------------------------------------------------------

def normalize_shap_values(
    shap_values: Any,
    class_index: int,
    feature_count: int,
) -> Array:
    """
    Normalize SHAP output across supported SHAP multiclass formats.

    SHAP versions may return:

    1. list[array] where each array is (samples, features)
    2. ndarray shaped (samples, features, classes)
    3. ndarray shaped (samples, classes, features)
    """
    if isinstance(shap_values, list):
        if class_index >= len(shap_values):
            raise IndexError(
                "SHAP class index is outside the returned class list."
            )

        values: Array = np.asarray(
            shap_values[class_index]
        )

        if values.ndim != 2:
            raise ValueError(
                "Expected class-specific SHAP values to have shape "
                "(samples, features). "
                f"Received {values.shape}"
            )

        return values

    values = np.asarray(shap_values)

    if values.ndim == 2:
        if values.shape[1] != feature_count:
            raise ValueError(
                "SHAP feature dimension does not match input features. "
                f"SHAP={values.shape}, "
                f"features={feature_count}"
            )

        return values

    if values.ndim != 3:
        raise ValueError(
            "Unsupported SHAP output shape: "
            f"{values.shape}"
        )

    # Format: (samples, features, classes)
    if values.shape[1] == feature_count:
        if class_index >= values.shape[2]:
            raise IndexError(
                "Class index exceeds SHAP class dimension."
            )

        return values[
            :,
            :,
            class_index,
        ]

    # Format: (samples, classes, features)
    if values.shape[2] == feature_count:
        if class_index >= values.shape[1]:
            raise IndexError(
                "Class index exceeds SHAP class dimension."
            )

        return values[
            :,
            class_index,
            :,
        ]

    raise ValueError(
        "Unable to determine SHAP feature/class dimensions. "
        f"Received shape {values.shape}, "
        f"expected feature count {feature_count}."
    )


def compute_error_shap(
    xgb_model: XGBClassifier,
    x_expanded: Array,
    y_true: Array,
    predictions: Array,
    true_class: int,
    predicted_class: int,
    output_name: str,
    sample_size: int = SHAP_SAMPLE_SIZE,
) -> None:
    """
    Calculate TreeSHAP importance for one confusion pair.

    Example
    -------
    true LOW -> predicted MODERATE

    Only observations belonging to the specified error pair are analyzed.
    """
    mask: Array = (
        (y_true == true_class)
        & (predictions == predicted_class)
    )

    indices: Array = np.flatnonzero(
        mask
    )

    pair_name = (
        f"{CLASS_NAMES[true_class]} -> "
        f"{CLASS_NAMES[predicted_class]}"
    )

    if indices.size == 0:
        print(
            f"  No observations for {pair_name}; skipping."
        )
        return

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    if indices.size > sample_size:
        indices = rng.choice(
            indices,
            size=sample_size,
            replace=False,
        )

    x_subset: Array = x_expanded[
        indices
    ]

    print(
        f"  SHAP samples: {x_subset.shape[0]}"
    )

    # -----------------------------------------------------------------------
    # IMPORTANT:
    # TreeExplainer receives the underlying XGBClassifier, NOT the custom
    # CongestionClassifier wrapper.
    # -----------------------------------------------------------------------
    explainer = shap.TreeExplainer(
        xgb_model
    )

    shap_values: Any = explainer.shap_values(
        x_subset
    )

    class_shap: Array = normalize_shap_values(
        shap_values=shap_values,
        class_index=predicted_class,
        feature_count=x_subset.shape[1],
    )

    mean_abs_shap: Array = np.mean(
        np.abs(class_shap),
        axis=0,
    )

    if mean_abs_shap.shape[0] != len(
        FEATURE_NAMES
    ):
        raise ValueError(
            "SHAP output feature count does not match FEATURE_NAMES. "
            f"SHAP={mean_abs_shap.shape[0]}, "
            f"names={len(FEATURE_NAMES)}"
        )

    result = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    result.to_csv(
        EXPLAINABILITY_DIR / output_name,
        index=False,
    )

    print(
        f"  Saved: "
        f"{EXPLAINABILITY_DIR / output_name}"
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_confidence_plot(
    probabilities: Array,
    predictions: Array,
    y_true: Array,
) -> None:
    """Plot confidence distributions for correct and incorrect predictions."""
    confidence: Array = calculate_confidence(
        probabilities
    )

    correct: Array = (
        predictions == y_true
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.hist(
        confidence[correct],
        bins=20,
        alpha=0.7,
        label="Correct",
    )

    plt.hist(
        confidence[~correct],
        bins=20,
        alpha=0.7,
        label="Incorrect",
    )

    plt.xlabel(
        "Prediction Confidence"
    )

    plt.ylabel(
        "Number of Samples"
    )

    plt.title(
        "Prediction Confidence: Correct vs Incorrect"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "confidence_correct_vs_incorrect.png",
        dpi=300,
    )

    plt.close()


def save_class_confidence_plot(
    confidence_by_class: pd.DataFrame,
) -> None:
    """Plot mean confidence by congestion class."""
    if confidence_by_class.empty:
        return

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        confidence_by_class["class"],
        confidence_by_class["mean_confidence"],
    )

    plt.xlabel(
        "True Congestion Class"
    )

    plt.ylabel(
        "Mean Prediction Confidence"
    )

    plt.title(
        "Prediction Confidence by Congestion Class"
    )

    plt.ylim(
        0.0,
        1.0,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "confidence_by_class.png",
        dpi=300,
    )

    plt.close()


def save_calibration_plot(
    calibration_df: pd.DataFrame,
) -> None:
    """Create a multiclass reliability diagram."""
    if calibration_df.empty:
        return

    plt.figure(
        figsize=(8, 7)
    )

    plt.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        label="Perfect Calibration",
    )

    plt.plot(
        calibration_df["mean_confidence"],
        calibration_df["accuracy"],
        marker="o",
        label="XGBoost",
    )

    plt.xlabel(
        "Mean Predicted Confidence"
    )

    plt.ylabel(
        "Observed Accuracy"
    )

    plt.title(
        "Congestion Classifier Reliability Diagram"
    )

    plt.xlim(
        0.0,
        1.0,
    )

    plt.ylim(
        0.0,
        1.0,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "calibration_reliability.png",
        dpi=300,
    )

    plt.close()


def save_error_confusion_plot(
    y_true: Array,
    predictions: Array,
) -> None:
    """Save normalized confusion matrix."""
    matrix: Array = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1, 2],
        normalize="true",
    )

    plt.figure(
        figsize=(8, 7)
    )

    plt.imshow(
        matrix
    )

    plt.colorbar()

    plt.xticks(
        range(3),
        CLASS_NAMES,
    )

    plt.yticks(
        range(3),
        CLASS_NAMES,
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    plt.title(
        "Normalized Congestion Classification Errors"
    )

    for row in range(3):
        for column in range(3):
            plt.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
            )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "error_confusion_analysis.png",
        dpi=300,
    )

    plt.close()


def save_sensor_error_plot(
    sensor_errors: pd.DataFrame,
) -> None:
    """Plot classification error rate for all sensors."""
    if sensor_errors.empty:
        return

    ordered: pd.DataFrame = sensor_errors.sort_values(
        "error_rate",
        ascending=False,
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        ordered["sensor_index"],
        ordered["error_rate"],
    )

    plt.xlabel(
        "Sensor Index"
    )

    plt.ylabel(
        "Classification Error Rate"
    )

    plt.title(
        "Sensor-Level Congestion Classification Error"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "sensor_error_distribution.png",
        dpi=300,
    )

    plt.close()


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute Sprint 4.7."""
    print("=" * 72)
    print(
        "SPRINT 4.7 — XAI ERROR ANALYSIS & CALIBRATION"
    )
    print("=" * 72)

    ensure_directories()

    # -----------------------------------------------------------------------
    # 1. Load
    # -----------------------------------------------------------------------

    print(
        "\n[1/8] Loading model and test data..."
    )

    model, x_test, y_test = load_data()

    print(
        f"X_test shape: {x_test.shape}"
    )

    print(
        f"y_test shape: {y_test.shape}"
    )

    print(
        f"Loaded model type: {type(model)}"
    )

    # -----------------------------------------------------------------------
    # 2. Expand
    # -----------------------------------------------------------------------

    print(
        "\n[2/8] Expanding sensor-level features..."
    )

    x_expanded, y_expanded = (
        expand_sensor_features(
            x_test,
            y_test,
        )
    )

    print(
        "Expanded feature matrix: "
        f"{x_expanded.shape}"
    )

    # -----------------------------------------------------------------------
    # 3. Predictions
    # -----------------------------------------------------------------------

    print(
        "\n[3/8] Generating predictions..."
    )

    predictions, probabilities = (
        generate_predictions(
            model,
            x_expanded,
        )
    )

    print(
        f"Predictions: {predictions.shape}"
    )

    print(
        f"Probabilities: {probabilities.shape}"
    )

    # -----------------------------------------------------------------------
    # 4. Confidence/error analysis
    # -----------------------------------------------------------------------

    print(
        "\n[4/8] Confidence and error analysis..."
    )

    confidence_by_class, error_analysis = (
        confidence_analysis(
            y_true=y_expanded,
            predictions=predictions,
            probabilities=probabilities,
        )
    )

    confidence_by_class.to_csv(
        METRICS_DIR
        / "xai_confidence_by_class.csv",
        index=False,
    )

    error_analysis.to_csv(
        METRICS_DIR
        / "xai_error_analysis.csv",
        index=False,
    )

    print(
        "\nConfidence by class:"
    )

    print(
        confidence_by_class.to_string(
            index=False
        )
    )

    print(
        "\nError matrix:"
    )

    print(
        error_analysis.to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------------
    # 5. Calibration
    # -----------------------------------------------------------------------

    print(
        "\n[5/8] Calibration analysis..."
    )

    ece, calibration_df = calculate_ece(
        y_true=y_expanded,
        probabilities=probabilities,
    )

    brier_score = (
        calculate_multiclass_brier_score(
            y_true=y_expanded,
            probabilities=probabilities,
            n_classes=len(CLASS_NAMES),
        )
    )

    calibration_df.to_csv(
        METRICS_DIR
        / "xai_calibration_bins.csv",
        index=False,
    )

    print(
        f"ECE: {ece:.6f}"
    )

    print(
        f"Multiclass Brier score: "
        f"{brier_score:.6f}"
    )

    # -----------------------------------------------------------------------
    # 6. Sensor-level analysis
    # -----------------------------------------------------------------------

    print(
        "\n[6/8] Sensor-level error analysis..."
    )

    sensor_errors = calculate_sensor_errors(
        y_test=y_test,
        predictions=predictions,
    )

    sensor_errors.to_csv(
        METRICS_DIR
        / "xai_sensor_error_analysis.csv",
        index=False,
    )

    print(
        "\nWorst 10 sensors by classification error:"
    )

    print(
        sensor_errors
        .sort_values(
            "error_rate",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    # -----------------------------------------------------------------------
    # 7. Error-specific SHAP
    # -----------------------------------------------------------------------

    print(
        "\n[7/8] Error-specific SHAP analysis..."
    )

    print(
        "Extracting underlying XGBoost estimator..."
    )

    xgb_model = extract_xgboost_classifier(
        model
    )

    print(
        "Underlying estimator: "
        f"{type(xgb_model)}"
    )

    error_pairs: list[
        tuple[int, int, str]
    ] = [
        (
            0,
            1,
            "shap_error_low_to_moderate.csv",
        ),
        (
            2,
            1,
            "shap_error_severe_to_moderate.csv",
        ),
        (
            1,
            0,
            "shap_error_moderate_to_low.csv",
        ),
        (
            1,
            2,
            "shap_error_moderate_to_severe.csv",
        ),
    ]

    for (
        true_class,
        predicted_class,
        filename,
    ) in error_pairs:
        print(
            f"\nAnalyzing "
            f"{CLASS_NAMES[true_class]} -> "
            f"{CLASS_NAMES[predicted_class]}"
        )

        compute_error_shap(
            xgb_model=xgb_model,
            x_expanded=x_expanded,
            y_true=y_expanded,
            predictions=predictions,
            true_class=true_class,
            predicted_class=predicted_class,
            output_name=filename,
        )

    # -----------------------------------------------------------------------
    # 8. Figures + summary
    # -----------------------------------------------------------------------

    print(
        "\n[8/8] Generating figures..."
    )

    save_confidence_plot(
        probabilities=probabilities,
        predictions=predictions,
        y_true=y_expanded,
    )

    save_class_confidence_plot(
        confidence_by_class=confidence_by_class,
    )

    save_calibration_plot(
        calibration_df=calibration_df,
    )

    save_error_confusion_plot(
        y_true=y_expanded,
        predictions=predictions,
    )

    save_sensor_error_plot(
        sensor_errors=sensor_errors,
    )

    confidence = calculate_confidence(
        probabilities
    )

    correct = (
        predictions == y_expanded
    )

    high_confidence = (
        confidence >= 0.8
    )

    low_confidence = (
        confidence < 0.5
    )

    accuracy = float(
        correct.mean()
    )

    high_confidence_fraction = float(
        high_confidence.mean()
    )

    low_confidence_fraction = float(
        low_confidence.mean()
    )

    high_confidence_accuracy: float | None

    if high_confidence.any():
        high_confidence_accuracy = float(
            correct[high_confidence].mean()
        )
    else:
        high_confidence_accuracy = None

    low_confidence_accuracy: float | None

    if low_confidence.any():
        low_confidence_accuracy = float(
            correct[low_confidence].mean()
        )
    else:
        low_confidence_accuracy = None

    summary: dict[str, Any] = {
        "experiment": "sprint_4.7",
        "model": "shared_xgboost_congestion_classifier",
        "explainer": "TreeSHAP",
        "evaluation_samples": int(
            len(y_expanded)
        ),
        "feature_count": int(
            x_expanded.shape[1]
        ),
        "classes": CLASS_NAMES,
        "mean_confidence": float(
            confidence.mean()
        ),
        "median_confidence": float(
            np.median(confidence)
        ),
        "accuracy": accuracy,
        "high_confidence_threshold": 0.8,
        "low_confidence_threshold": 0.5,
        "high_confidence_fraction": (
            high_confidence_fraction
        ),
        "low_confidence_fraction": (
            low_confidence_fraction
        ),
        "high_confidence_accuracy": (
            high_confidence_accuracy
        ),
        "low_confidence_accuracy": (
            low_confidence_accuracy
        ),
        "ece": float(ece),
        "multiclass_brier_score": float(
            brier_score
        ),
        "calibration_bins": int(
            len(calibration_df)
        ),
    }

    summary_path: Final[Path] = (
        METRICS_DIR
        / "xai_calibration_metrics.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        "\n" + "=" * 72
    )

    print(
        "SPRINT 4.7 COMPLETED"
    )

    print(
        "=" * 72
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Mean confidence: "
        f"{confidence.mean():.4f}"
    )

    print(
        f"High-confidence fraction: "
        f"{high_confidence_fraction:.4f}"
    )

    print(
        f"High-confidence accuracy: "
        f"{high_confidence_accuracy}"
    )

    print(
        f"Low-confidence fraction: "
        f"{low_confidence_fraction:.4f}"
    )

    print(
        f"Low-confidence accuracy: "
        f"{low_confidence_accuracy}"
    )

    print(
        f"ECE: {ece:.6f}"
    )

    print(
        "Multiclass Brier score: "
        f"{brier_score:.6f}"
    )

    print(
        f"\nSummary saved to:\n{summary_path}"
    )


if __name__ == "__main__":
    main()
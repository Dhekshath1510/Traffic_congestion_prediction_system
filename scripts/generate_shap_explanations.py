from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# ============================================================================
# Project paths
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BACKEND_ROOT = PROJECT_ROOT / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from app.ml.explainability.shap_explainer import (  # noqa: E402
    CLASS_NAMES,
    CongestionSHAPExplainer,
)


# ============================================================================
# Paths
# ============================================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "congestion_classifier.joblib"
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "congestion"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

METRICS_DIR = (
    RESULTS_DIR
    / "metrics"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

XAI_DIR = (
    RESULTS_DIR
    / "explainability"
)

for directory in (
    METRICS_DIR,
    FIGURES_DIR,
    XAI_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# Configuration
# ============================================================================

# SHAP does not need the entire 1M+ observation training dataset.
# A statistically useful sample is sufficient for global explanation.

SHAP_SAMPLE_SIZE = 10_000

RANDOM_STATE = 42

TOP_FEATURES = 15

INSTANCE_INDEX = 0


# ============================================================================
# Feature definitions
# ============================================================================

FEATURE_NAMES = [
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


# ============================================================================
# Loading
# ============================================================================


def load_model():
    """Load the trained congestion classifier."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Classifier model not found: {MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    if not hasattr(
        model,
        "predict",
    ):
        raise TypeError(
            "Loaded artifact does not appear to be "
            "a congestion classifier."
        )

    return model


def load_test_features() -> np.ndarray:
    """Load test features and expand them with sensor identity."""

    path = DATA_DIR / "X_test.npy"

    if not path.exists():
        raise FileNotFoundError(
            f"Test features not found: {path}"
        )

    X = np.load(
        path,
        allow_pickle=False,
    )

    if X.ndim != 2:
        raise ValueError(
            f"Expected 2-D test features, got {X.shape}"
        )

    # Sprint 4.3 stores 24 engineered features.
    expected_base_features = 24

    if X.shape[1] != expected_base_features:
        raise ValueError(
            "Unexpected number of base features: "
            f"{X.shape[1]} != {expected_base_features}"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Test features contain NaN or infinite values."
        )

    # The classifier was trained on one observation per
    # (timestamp, sensor) pair.
    #
    # Therefore the same expansion used during Sprint 4.4
    # must be reproduced here.
    sensor_count = 207

    samples = X.shape[0]

    expanded_features = np.repeat(
        X,
        sensor_count,
        axis=0,
    )

    sensor_indices = np.tile(
        np.arange(sensor_count),
        samples,
    ).astype(
        np.float32
    )

    expanded_X = np.column_stack(
        [
            expanded_features,
            sensor_indices,
        ]
    ).astype(
        np.float32,
        copy=False,
    )

    if expanded_X.shape[1] != 25:
        raise RuntimeError(
            "Expanded SHAP features do not match "
            "the classifier input dimension: "
            f"{expanded_X.shape[1]} != 25"
        )

    print(
        f"Base test features:     {X.shape}"
    )

    print(
        f"Expanded test features: {expanded_X.shape}"
    )

    return expanded_X

def sample_for_shap(
    X: np.ndarray,
) -> np.ndarray:
    """Create a reproducible SHAP analysis sample."""

    if len(X) <= SHAP_SAMPLE_SIZE:
        return X

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    indices = rng.choice(
        len(X),
        size=SHAP_SAMPLE_SIZE,
        replace=False,
    )

    indices.sort()

    return X[indices]


# ============================================================================
# Visualization
# ============================================================================


def save_global_importance_plot(
    importance: pd.DataFrame,
) -> None:
    """Create the global mean-absolute SHAP plot."""

    top = importance.head(
        TOP_FEATURES
    ).sort_values(
        "mean_abs_shap"
    )

    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    axis.barh(
        top["feature"],
        top["mean_abs_shap"],
    )

    axis.set_xlabel(
        "Mean absolute SHAP value"
    )

    axis.set_ylabel(
        "Feature"
    )

    axis.set_title(
        "Global Feature Importance for Congestion Classification"
    )

    figure.tight_layout()

    output = (
        FIGURES_DIR
        / "shap_global_feature_importance.png"
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


def save_class_importance_plot(
    importance: pd.DataFrame,
    class_name: str,
) -> None:
    """Create class-specific SHAP importance plot."""

    top = importance.head(
        TOP_FEATURES
    ).sort_values(
        "mean_abs_shap"
    )

    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    axis.barh(
        top["feature"],
        top["mean_abs_shap"],
    )

    axis.set_xlabel(
        "Mean absolute SHAP value"
    )

    axis.set_ylabel(
        "Feature"
    )

    axis.set_title(
        f"SHAP Feature Importance — {class_name}"
    )

    figure.tight_layout()

    filename = (
        "shap_"
        f"{class_name.lower()}"
        "_importance.png"
    )

    output = (
        FIGURES_DIR / filename
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


def save_summary_plot(
    explainer: CongestionSHAPExplainer,
    X: np.ndarray,
) -> None:
    """
    Generate a SHAP summary visualization.

    A separate summary plot is generated for each congestion class.
    """

    shap_values = explainer.calculate(
        X
    )

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        figure = plt.figure(
            figsize=(10, 7)
        )

        shap.summary_plot(
            shap_values[:, :, class_id],
            X,
            feature_names=FEATURE_NAMES,
            max_display=TOP_FEATURES,
            show=False,
        )

        plt.title(
            f"SHAP Summary — {class_name}"
        )

        plt.tight_layout()

        output = (
            FIGURES_DIR
            / "shap_summary_"
            f"{class_name.lower()}.png"
        )

        plt.savefig(
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


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run Sprint 4.5 SHAP analysis."""

    print("=" * 64)
    print("Sprint 4.5 — SHAP Explainability")
    print("=" * 64)

    # ------------------------------------------------------------------------
    # Load model and data
    # ------------------------------------------------------------------------

    print(
        "\nLoading trained classifier..."
    )

    model = load_model()

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        "\nLoading test features..."
    )

    X_test = load_test_features()

    print(
        f"Test features: {X_test.shape}"
    )

    # ------------------------------------------------------------------------
    # Create SHAP sample
    # ------------------------------------------------------------------------

    X_shap = sample_for_shap(
        X_test
    )

    print(
        f"SHAP sample: {X_shap.shape}"
    )

    # ------------------------------------------------------------------------
    # Create explainer
    # ------------------------------------------------------------------------

    print(
        "\nInitializing TreeSHAP..."
    )

    explainer = CongestionSHAPExplainer(
        model,
        FEATURE_NAMES,
    )

    # ------------------------------------------------------------------------
    # Global importance
    # ------------------------------------------------------------------------

    print(
        "\nCalculating global SHAP importance..."
    )

    global_importance = (
        explainer.global_importance(
            X_shap
        )
    )

    global_path = (
        XAI_DIR
        / "shap_global_importance.csv"
    )

    global_importance.to_csv(
        global_path,
        index=False,
    )

    print(
        f"Saved: {global_path}"
    )

    print(
        "\nTop global features"
    )

    print(
        "==================="
    )

    print(
        global_importance.head(
            TOP_FEATURES
        ).to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    save_global_importance_plot(
        global_importance
    )

    # ------------------------------------------------------------------------
    # Class-specific importance
    # ------------------------------------------------------------------------

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"\nCalculating SHAP importance for {class_name}..."
        )

        importance = (
            explainer.class_importance(
                X_shap,
                class_id,
            )
        )

        path = (
            XAI_DIR
            / "shap_"
            f"{class_name.lower()}"
            "_importance.csv"
        )

        importance.to_csv(
            path,
            index=False,
        )

        print(
            f"Saved: {path}"
        )

        print(
            importance.head(
                TOP_FEATURES
            ).to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.6f}"
                ),
            )
        )

        save_class_importance_plot(
            importance,
            class_name,
        )

    # ------------------------------------------------------------------------
    # SHAP summary plots
    # ------------------------------------------------------------------------

    print(
        "\nGenerating SHAP summary plots..."
    )

    save_summary_plot(
        explainer,
        X_shap,
    )

    # ------------------------------------------------------------------------
    # Individual explanation
    # ------------------------------------------------------------------------

    print(
        "\nGenerating individual prediction explanation..."
    )

    explanation = (
        explainer.explain_instance(
            X_test,
            instance_index=INSTANCE_INDEX,
            top_n=10,
        )
    )

    explanation_dict = {
        "predicted_class": (
            explanation.predicted_class
        ),
        "predicted_class_id": (
            explanation.predicted_class_id
        ),
        "probabilities": (
            explanation.probabilities
        ),
        "feature_contributions": (
            explanation.feature_contributions
        ),
    }

    explanation_path = (
        XAI_DIR
        / "example_prediction_explanation.json"
    )

    explanation_path.write_text(
        json.dumps(
            explanation_dict,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved: {explanation_path}"
    )

    print(
        "\nExample prediction"
    )

    print(
        "=================="
    )

    print(
        f"Predicted class: "
        f"{explanation.predicted_class}"
    )

    print(
        "\nClass probabilities:"
    )

    for (
        class_name,
        probability,
    ) in explanation.probabilities.items():

        print(
            f"{class_name:<10}: "
            f"{probability:.4f}"
        )

    print(
        "\nTop contributing features:"
    )

    for contribution in (
        explanation.feature_contributions
    ):
        print(
            f"{contribution['feature']:<30} "
            f"value={contribution['value']:.4f} "
            f"SHAP={contribution['shap_value']:+.6f} "
            f"({contribution['direction']})"
        )

    # ------------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------------

    metadata = {
        "experiment": "sprint_4.5",
        "explainer": "TreeSHAP",
        "model": (
            "shared_xgboost_congestion_classifier"
        ),
        "classes": list(
            CLASS_NAMES
        ),
        "feature_count": len(
            FEATURE_NAMES
        ),
        "feature_names": FEATURE_NAMES,
        "shap_sample_size": len(
            X_shap
        ),
        "random_state": RANDOM_STATE,
        "aggregation": (
            "mean absolute SHAP across samples "
            "and classes for global importance"
        ),
        "individual_explanation": {
            "test_instance": INSTANCE_INDEX,
            "top_features": 10,
        },
    }

    metadata_path = (
        XAI_DIR
        / "shap_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nSaved: {metadata_path}"
    )

    print(
        "\n" + "=" * 64
    )

    print(
        "Sprint 4.5 completed."
    )

    print(
        "=" * 64
    )


if __name__ == "__main__":
    main()
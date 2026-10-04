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
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BACKEND_ROOT = PROJECT_ROOT / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from app.ml.explainability.shap_explainer import (  # noqa: E402
    CLASS_NAMES,
    CongestionSHAPExplainer,
)


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

RESULTS_DIR = PROJECT_ROOT / "results"

XAI_DIR = RESULTS_DIR / "explainability"

FIGURES_DIR = RESULTS_DIR / "figures"

METRICS_DIR = RESULTS_DIR / "metrics"


for directory in (
    XAI_DIR,
    FIGURES_DIR,
    METRICS_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_STATE = 42

# Number of observations used for the XAI evaluation.
EVALUATION_SAMPLE_SIZE = 10_000

# Number of observations used for representative examples.
REPRESENTATIVE_SAMPLE_SIZE = 500

TOP_N_FEATURES = 10

# Confidence thresholds used only for analysis.
HIGH_CONFIDENCE_THRESHOLD = 0.80
LOW_CONFIDENCE_THRESHOLD = 0.50


# ============================================================================
# FEATURES
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
# DATA LOADING
# ============================================================================


def load_test_features() -> np.ndarray:
    """
    Load Sprint 4.3 test features and reproduce the exact sensor expansion
    used during Sprint 4.4 classifier training.
    """

    path = DATA_DIR / "X_test.npy"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing test feature file: {path}"
        )

    X = np.load(
        path,
        allow_pickle=False,
    )

    if X.ndim != 2:
        raise ValueError(
            f"Expected 2-D X_test, got {X.shape}"
        )

    if X.shape[1] != 24:
        raise ValueError(
            "Expected 24 base features, "
            f"got {X.shape[1]}"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "X_test contains NaN or infinite values."
        )

    sensor_count = 207

    samples = X.shape[0]

    expanded = np.repeat(
        X,
        sensor_count,
        axis=0,
    )

    sensor_index = np.tile(
        np.arange(sensor_count),
        samples,
    ).astype(
        np.float32
    )

    expanded_X = np.column_stack(
        [
            expanded,
            sensor_index,
        ]
    ).astype(
        np.float32,
        copy=False,
    )

    if expanded_X.shape[1] != len(
        FEATURE_NAMES
    ):
        raise RuntimeError(
            "Expanded feature count does not match "
            "FEATURE_NAMES."
        )

    return expanded_X


def load_test_labels() -> np.ndarray:
    """
    Load Sprint 4.3 test labels and flatten them to sensor observations.
    """

    path = DATA_DIR / "y_test.npy"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing test labels: {path}"
        )

    y = np.load(
        path,
        allow_pickle=False,
    )

    if y.ndim != 2:
        raise ValueError(
            f"Expected 2-D y_test, got {y.shape}"
        )

    return y.reshape(
        -1
    ).astype(
        np.int8,
        copy=False,
    )


# ============================================================================
# SAMPLING
# ============================================================================


def sample_indices(
    size: int,
    sample_size: int,
) -> np.ndarray:
    """Create reproducible random sample indices."""

    if sample_size >= size:
        return np.arange(size)

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    indices = rng.choice(
        size,
        size=sample_size,
        replace=False,
    )

    indices.sort()

    return indices


# ============================================================================
# FEATURE STABILITY
# ============================================================================


def calculate_top_feature_frequency(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int,
) -> pd.DataFrame:
    """
    Calculate how frequently each feature appears among the top-N
    absolute SHAP contributors.
    """

    sample_count = shap_values.shape[0]

    counts = np.zeros(
        len(feature_names),
        dtype=np.int64,
    )

    for sample in shap_values:
        absolute_values = np.abs(
            sample
        )

        top_indices = np.argsort(
            absolute_values
        )[-top_n:]

        counts[top_indices] += 1

    frequency = (
        counts / sample_count
    )

    dataframe = pd.DataFrame(
        {
            "feature": feature_names,
            "top_n_frequency": frequency,
            "top_n_count": counts,
        }
    )

    return dataframe.sort_values(
        "top_n_frequency",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ============================================================================
# CLASS-SPECIFIC STABILITY
# ============================================================================


def calculate_class_stability(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Calculate class-specific mean absolute SHAP values.
    """

    rows = []

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        values = shap_values[
            :,
            :,
            class_id,
        ]

        importance = np.mean(
            np.abs(values),
            axis=0,
        )

        for feature, value in zip(
            feature_names,
            importance,
            strict=True,
        ):
            rows.append(
                {
                    "class_id": class_id,
                    "class": class_name,
                    "feature": feature,
                    "mean_abs_shap": float(
                        value
                    ),
                }
            )

    dataframe = pd.DataFrame(
        rows
    )

    return dataframe


# ============================================================================
# CONFIDENCE ANALYSIS
# ============================================================================


def calculate_confidence_analysis(
    model,
    X: np.ndarray,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    """
    Relate prediction confidence to SHAP explanation magnitude.

    Explanation strength is represented by the sum of absolute SHAP values
    for the predicted class.
    """

    predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )

    rows = []

    for index in range(
        len(X)
    ):
        predicted_class = int(
            predictions[index]
        )

        confidence = float(
            probabilities[
                index,
                predicted_class,
            ]
        )

        explanation_strength = float(
            np.sum(
                np.abs(
                    shap_values[
                        index,
                        :,
                        predicted_class,
                    ]
                )
            )
        )

        rows.append(
            {
                "sample_index": index,
                "predicted_class_id": predicted_class,
                "predicted_class": CLASS_NAMES[
                    predicted_class
                ],
                "confidence": confidence,
                "explanation_strength": (
                    explanation_strength
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# REPRESENTATIVE EXPLANATIONS
# ============================================================================


def select_representative_examples(
    model,
    X: np.ndarray,
    shap_values: np.ndarray,
) -> list[dict]:
    """
    Select one representative prediction for each class.

    Preference:
        high-confidence predictions with strong explanations.
    """

    predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )

    examples = []

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        candidate_indices = np.where(
            predictions == class_id
        )[0]

        if len(candidate_indices) == 0:
            continue

        best_index = None
        best_score = -np.inf

        for index in candidate_indices:
            confidence = probabilities[
                index,
                class_id,
            ]

            explanation_strength = np.sum(
                np.abs(
                    shap_values[
                        index,
                        :,
                        class_id,
                    ]
                )
            )

            score = (
                float(confidence)
                * float(explanation_strength)
            )

            if score > best_score:
                best_score = score
                best_index = int(
                    index
                )

        if best_index is None:
            continue

        class_shap = shap_values[
            best_index,
            :,
            class_id,
        ]

        absolute_shap = np.abs(
            class_shap
        )

        top_indices = np.argsort(
            absolute_shap
        )[-TOP_N_FEATURES:][::-1]

        factors = []

        for feature_index in top_indices:
            factors.append(
                {
                    "feature": FEATURE_NAMES[
                        feature_index
                    ],
                    "shap_value": float(
                        class_shap[
                            feature_index
                        ]
                    ),
                    "direction": (
                        "toward_class"
                        if class_shap[
                            feature_index
                        ] > 0
                        else "away_from_class"
                    ),
                }
            )

        examples.append(
            {
                "class_id": class_id,
                "class": class_name,
                "sample_index": best_index,
                "confidence": float(
                    probabilities[
                        best_index,
                        class_id,
                    ]
                ),
                "top_factors": factors,
            }
        )

    return examples


# ============================================================================
# VISUALIZATIONS
# ============================================================================


def save_top_feature_frequency_plot(
    frequency: pd.DataFrame,
) -> None:
    """Visualize the frequency of top SHAP contributors."""

    top = frequency.head(
        15
    ).sort_values(
        "top_n_frequency"
    )

    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    axis.barh(
        top["feature"],
        top["top_n_frequency"],
    )

    axis.set_xlabel(
        f"Fraction appearing in top-{TOP_N_FEATURES} contributors"
    )

    axis.set_ylabel(
        "Feature"
    )

    axis.set_title(
        "Stability of SHAP Feature Contributions"
    )

    figure.tight_layout()

    output = (
        FIGURES_DIR
        / "shap_feature_contribution_frequency.png"
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


def save_confidence_strength_plot(
    dataframe: pd.DataFrame,
) -> None:
    """Visualize prediction confidence against explanation strength."""

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        subset = dataframe[
            dataframe[
                "predicted_class_id"
            ]
            == class_id
        ]

        if subset.empty:
            continue

        axis.scatter(
            subset["confidence"],
            subset["explanation_strength"],
            label=class_name,
            alpha=0.5,
            s=15,
        )

    axis.set_xlabel(
        "Prediction confidence"
    )

    axis.set_ylabel(
        "Explanation strength"
    )

    axis.set_title(
        "Prediction Confidence vs. SHAP Explanation Strength"
    )

    axis.legend()

    figure.tight_layout()

    output = (
        FIGURES_DIR
        / "shap_confidence_vs_explanation_strength.png"
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


def save_class_importance_comparison(
    class_stability: pd.DataFrame,
) -> None:
    """
    Plot the strongest features for each congestion class.

    Each class gets its own figure to keep the visualization readable.
    """

    for class_name in CLASS_NAMES:
        subset = class_stability[
            class_stability["class"]
            == class_name
        ].sort_values(
            "mean_abs_shap",
            ascending=False,
        ).head(
            10
        ).sort_values(
            "mean_abs_shap"
        )

        figure, axis = plt.subplots(
            figsize=(9, 6)
        )

        axis.barh(
            subset["feature"],
            subset["mean_abs_shap"],
        )

        axis.set_xlabel(
            "Mean absolute SHAP value"
        )

        axis.set_ylabel(
            "Feature"
        )

        axis.set_title(
            f"Class-Specific SHAP Importance — {class_name}"
        )

        figure.tight_layout()

        output = (
            FIGURES_DIR
            / "xai_evaluation_"
            f"{class_name.lower()}"
            "_importance.png"
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


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """Run Sprint 4.6 XAI evaluation."""

    print("=" * 64)
    print("Sprint 4.6 — Explainable Congestion Evaluation")
    print("=" * 64)

    # ------------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------------

    print(
        "\nLoading classifier..."
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Classifier not found: {MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    # ------------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------------

    print(
        "Loading test data..."
    )

    X_test = load_test_features()

    y_test = load_test_labels()

    print(
        f"Test features: {X_test.shape}"
    )

    print(
        f"Test labels:   {y_test.shape}"
    )

    if len(X_test) != len(y_test):
        raise RuntimeError(
            "Test feature/label counts do not match."
        )

    # ------------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------------

    indices = sample_indices(
        len(X_test),
        EVALUATION_SAMPLE_SIZE,
    )

    X_eval = X_test[
        indices
    ]

    y_eval = y_test[
        indices
    ]

    print(
        f"Evaluation sample: {X_eval.shape}"
    )

    # ------------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------------

    print(
        "\nCalculating SHAP values..."
    )

    explainer = CongestionSHAPExplainer(
        model,
        FEATURE_NAMES,
    )

    shap_values = explainer.calculate(
        X_eval
    )

    print(
        f"SHAP shape: {shap_values.shape}"
    )

    # ------------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------------

    predictions = model.predict(
        X_eval
    )

    probabilities = model.predict_proba(
        X_eval
    )

    # ------------------------------------------------------------------------
    # Global importance
    # ------------------------------------------------------------------------

    global_importance = (
        explainer.global_importance(
            X_eval
        )
    )

    global_path = (
        XAI_DIR
        / "xai_evaluation_global_importance.csv"
    )

    global_importance.to_csv(
        global_path,
        index=False,
    )

    print(
        f"\nSaved: {global_path}"
    )

    # ------------------------------------------------------------------------
    # Feature frequency
    # ------------------------------------------------------------------------

    print(
        "\nCalculating contribution stability..."
    )

    frequency_rows = []

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        class_mask = (
            predictions == class_id
        )

        if not np.any(
            class_mask
        ):
            continue

        class_values = shap_values[
            class_mask,
            :,
            class_id,
        ]

        frequency = (
            calculate_top_feature_frequency(
                class_values,
                FEATURE_NAMES,
                TOP_N_FEATURES,
            )
        )

        frequency.insert(
            0,
            "class_id",
            class_id,
        )

        frequency.insert(
            1,
            "class",
            class_name,
        )

        frequency_rows.append(
            frequency
        )

    if frequency_rows:
        feature_frequency = pd.concat(
            frequency_rows,
            ignore_index=True,
        )
    else:
        feature_frequency = pd.DataFrame()

    frequency_path = (
        XAI_DIR
        / "xai_feature_contribution_frequency.csv"
    )

    feature_frequency.to_csv(
        frequency_path,
        index=False,
    )

    print(
        f"Saved: {frequency_path}"
    )

    # ------------------------------------------------------------------------
    # Class stability
    # ------------------------------------------------------------------------

    class_stability = (
        calculate_class_stability(
            shap_values,
            FEATURE_NAMES,
        )
    )

    class_path = (
        XAI_DIR
        / "xai_class_feature_importance.csv"
    )

    class_stability.to_csv(
        class_path,
        index=False,
    )

    print(
        f"Saved: {class_path}"
    )

    # ------------------------------------------------------------------------
    # Confidence analysis
    # ------------------------------------------------------------------------

    confidence_dataframe = (
        calculate_confidence_analysis(
            model,
            X_eval,
            shap_values,
        )
    )

    confidence_path = (
        XAI_DIR
        / "xai_confidence_analysis.csv"
    )

    confidence_dataframe.to_csv(
        confidence_path,
        index=False,
    )

    print(
        f"Saved: {confidence_path}"
    )

    # ------------------------------------------------------------------------
    # Confidence statistics
    # ------------------------------------------------------------------------

    confidence_summary = []

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        subset = confidence_dataframe[
            confidence_dataframe[
                "predicted_class_id"
            ]
            == class_id
        ]

        if subset.empty:
            continue

        confidence_summary.append(
            {
                "class_id": class_id,
                "class": class_name,
                "count": int(
                    len(subset)
                ),
                "mean_confidence": float(
                    subset["confidence"].mean()
                ),
                "median_confidence": float(
                    subset["confidence"].median()
                ),
                "high_confidence_fraction": float(
                    (
                        subset["confidence"]
                        >= HIGH_CONFIDENCE_THRESHOLD
                    ).mean()
                ),
                "mean_explanation_strength": float(
                    subset[
                        "explanation_strength"
                    ].mean()
                ),
            }
        )

    confidence_summary_df = pd.DataFrame(
        confidence_summary
    )

    confidence_summary_path = (
        XAI_DIR
        / "xai_confidence_summary.csv"
    )

    confidence_summary_df.to_csv(
        confidence_summary_path,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Representative examples
    # ------------------------------------------------------------------------

    print(
        "\nSelecting representative examples..."
    )

    representative = (
        select_representative_examples(
            model,
            X_eval,
            shap_values,
        )
    )

    representative_path = (
        XAI_DIR
        / "representative_explanations.json"
    )

    representative_path.write_text(
        json.dumps(
            representative,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved: {representative_path}"
    )

    # ------------------------------------------------------------------------
    # Print representative examples
    # ------------------------------------------------------------------------

    print(
        "\nRepresentative predictions"
    )

    print(
        "==========================="
    )

    for example in representative:

        print(
            f"\n{example['class']}"
        )

        print(
            f"Confidence: "
            f"{example['confidence']:.4f}"
        )

        print(
            "Top factors:"
        )

        for factor in example[
            "top_factors"
        ]:
            print(
                f"  {factor['feature']:<30} "
                f"{factor['shap_value']:+.6f} "
                f"{factor['direction']}"
            )

    # ------------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------------

    print(
        "\nGenerating evaluation figures..."
    )

    save_top_feature_frequency_plot(
        feature_frequency[
            feature_frequency["class"]
            == "SEVERE"
        ]
        if not feature_frequency.empty
        else pd.DataFrame(
            columns=[
                "feature",
                "top_n_frequency",
            ]
        )
    )

    save_confidence_strength_plot(
        confidence_dataframe
    )

    save_class_importance_comparison(
        class_stability
    )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    summary = {
        "experiment": "sprint_4.6",
        "model": (
            "shared_xgboost_congestion_classifier"
        ),
        "explainer": "TreeSHAP",
        "evaluation_sample_size": int(
            len(X_eval)
        ),
        "feature_count": len(
            FEATURE_NAMES
        ),
        "classes": list(
            CLASS_NAMES
        ),
        "high_confidence_threshold": (
            HIGH_CONFIDENCE_THRESHOLD
        ),
        "low_confidence_threshold": (
            LOW_CONFIDENCE_THRESHOLD
        ),
        "top_n_features": TOP_N_FEATURES,
        "prediction_distribution": {
            CLASS_NAMES[class_id]: int(
                np.sum(
                    predictions == class_id
                )
            )
            for class_id in range(3)
        },
        "actual_distribution": {
            CLASS_NAMES[class_id]: int(
                np.sum(
                    y_eval == class_id
                )
            )
            for class_id in range(3)
        },
        "mean_confidence": float(
            confidence_dataframe[
                "confidence"
            ].mean()
        ),
        "median_confidence": float(
            confidence_dataframe[
                "confidence"
            ].median()
        ),
        "mean_explanation_strength": float(
            confidence_dataframe[
                "explanation_strength"
            ].mean()
        ),
    }

    summary_path = (
        XAI_DIR
        / "xai_evaluation_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nSaved: {summary_path}"
    )

    print(
        "\n" + "=" * 64
    )

    print(
        "Sprint 4.6 completed."
    )

    print(
        "=" * 64
    )


if __name__ == "__main__":
    main()
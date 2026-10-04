from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

BACKEND_ROOT = PROJECT_ROOT / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences"
)

RESULTS_DIR = PROJECT_ROOT / "results"

PREDICTION_DIR = (
    RESULTS_DIR
    / "predictions"
)

METRICS_DIR = (
    RESULTS_DIR
    / "metrics"
)

FORECASTING_DIR = (
    RESULTS_DIR
    / "forecasting"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)


# ---------------------------------------------------------------------------
# Experiment results already obtained
# ---------------------------------------------------------------------------

BASELINE_RESULTS = [
    {
        "model_name": "Persistence",
        "split": "Validation",
        "mae": 4.0593,
        "rmse": 9.9222,
        "smape": 11.18,
    },
    {
        "model_name": "XGBoost",
        "split": "Validation",
        "mae": 4.6567,
        "rmse": 9.5430,
        "smape": 24.93,
    },
    {
        "model_name": "GRU-v1",
        "split": "Validation",
        "mae": 10.0922,
        "rmse": 15.5644,
        "smape": 34.71,
    },
    {
        "model_name": "GRU-v2",
        "split": "Validation",
        "mae": 4.0459,
        "rmse": 9.8186,
        "smape": 25.15,
    },
    {
        "model_name": "Persistence",
        "split": "Test",
        "mae": 3.8795,
        "rmse": 9.5040,
        "smape": 11.02,
    },
    {
        "model_name": "XGBoost",
        "split": "Test",
        "mae": 4.6272,
        "rmse": 9.2205,
        "smape": 32.77,
    },
    {
        "model_name": "GRU-v1",
        "split": "Test",
        "mae": 9.7395,
        "rmse": 15.5530,
        "smape": 41.49,
    },
    {
        "model_name": "GRU-v2",
        "split": "Test",
        "mae": 3.9040,
        "rmse": 9.4304,
        "smape": 33.01,
    },
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def ensure_directories() -> None:
    """Create result directories when they do not exist."""

    FORECASTING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_array(filename: str) -> np.ndarray:
    """Load a prepared sequence artifact."""

    path = SEQUENCE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required file does not exist: {path}"
        )

    array = np.load(path)

    if not np.isfinite(array).all():
        raise ValueError(
            f"File contains NaN or infinite values: {path}"
        )

    return array.astype(
        np.float32,
        copy=False,
    )


def calculate_mae(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate global mean absolute error."""

    return float(
        np.mean(
            np.abs(
                actual - predicted
            )
        )
    )


def calculate_rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate global root mean squared error."""

    return float(
        np.sqrt(
            np.mean(
                np.square(
                    actual - predicted
                )
            )
        )
    )


def calculate_smape(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate symmetric mean absolute percentage error."""

    denominator = (
        np.abs(actual)
        + np.abs(predicted)
    )

    numerator = (
        2.0
        * np.abs(
            actual - predicted
        )
    )

    ratio = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(
            numerator,
            dtype=np.float32,
        ),
        where=denominator != 0,
    )

    return float(
        np.mean(ratio) * 100.0
    )


# ---------------------------------------------------------------------------
# Prediction-level analysis
# ---------------------------------------------------------------------------


def evaluate_gru_v2(
    actual: np.ndarray,
    predicted: np.ndarray,
    split: str,
) -> dict[str, float | str]:
    """Evaluate the saved GRU-v2 predictions independently."""

    if actual.shape != predicted.shape:
        raise ValueError(
            f"{split}: actual and predicted shapes differ: "
            f"{actual.shape} != {predicted.shape}"
        )

    return {
        "model_name": "GRU-v2",
        "split": split,
        "mae": calculate_mae(
            actual,
            predicted,
        ),
        "rmse": calculate_rmse(
            actual,
            predicted,
        ),
        "smape": calculate_smape(
            actual,
            predicted,
        ),
    }


def calculate_per_sensor_mae(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    """Calculate MAE independently for each of the 207 sensors."""

    return np.mean(
        np.abs(
            actual - predicted
        ),
        axis=0,
    )


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def save_model_comparison(
    results: pd.DataFrame,
) -> None:
    """Generate model-level MAE/RMSE/sMAPE comparison."""

    long_results = results.melt(
        id_vars=[
            "model_name",
            "split",
        ],
        value_vars=[
            "mae",
            "rmse",
            "smape",
        ],
        var_name="metric",
        value_name="value",
    )

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    sns.barplot(
        data=long_results,
        x="model_name",
        y="value",
        hue="split",
        ax=ax,
    )

    ax.set_title(
        "Forecasting Performance Comparison"
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Metric value")

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "forecasting_model_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_gru_actual_vs_predicted(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> None:
    """Plot actual and GRU-v2 predicted network speed."""

    actual_network = np.median(
        actual,
        axis=1,
    )

    predicted_network = np.median(
        predicted,
        axis=1,
    )

    sample_count = min(
        1000,
        len(actual_network),
    )

    indices = np.arange(
        sample_count
    )

    fig, ax = plt.subplots(
        figsize=(10, 4.5)
    )

    ax.plot(
        indices,
        actual_network[:sample_count],
        label="Actual",
        linewidth=1.2,
    )

    ax.plot(
        indices,
        predicted_network[:sample_count],
        label="GRU-v2",
        linewidth=1.2,
    )

    ax.set_title(
        "Actual vs Predicted Network Traffic Speed"
    )

    ax.set_xlabel(
        "Test sample"
    )

    ax.set_ylabel(
        "Median traffic speed"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "gru_actual_vs_predicted.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_residual_distribution(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> None:
    """Generate the GRU-v2 residual distribution."""

    residuals = (
        actual - predicted
    ).ravel()

    # Keep plotting manageable while retaining
    # reproducibility.
    if len(residuals) > 500_000:
        rng = np.random.default_rng(42)

        indices = rng.choice(
            len(residuals),
            size=500_000,
            replace=False,
        )

        residuals = residuals[indices]

    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    sns.histplot(
        residuals,
        bins=50,
        kde=True,
        ax=ax,
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_title(
        "GRU-v2 Prediction Residual Distribution"
    )

    ax.set_xlabel(
        "Residual (actual - predicted)"
    )

    ax.set_ylabel(
        "Frequency"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "gru_residual_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_per_sensor_mae(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> pd.DataFrame:
    """Generate and save sensor-level GRU-v2 MAE."""

    sensor_mae = calculate_per_sensor_mae(
        actual,
        predicted,
    )

    sensor_results = pd.DataFrame(
        {
            "sensor_index": np.arange(
                1,
                len(sensor_mae) + 1,
            ),
            "mae": sensor_mae,
        }
    )

    sensor_results.to_csv(
        FORECASTING_DIR
        / "per_sensor_mae.csv",
        index=False,
    )

    fig, ax = plt.subplots(
        figsize=(10, 4.5)
    )

    sns.lineplot(
        data=sensor_results,
        x="sensor_index",
        y="mae",
        linewidth=1.0,
        ax=ax,
    )

    ax.set_title(
        "GRU-v2 MAE Across 207 Traffic Sensors"
    )

    ax.set_xlabel(
        "Sensor index"
    )

    ax.set_ylabel(
        "MAE"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "gru_per_sensor_mae.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return sensor_results


def save_error_distribution_by_split(
    validation_actual: np.ndarray,
    validation_predicted: np.ndarray,
    test_actual: np.ndarray,
    test_predicted: np.ndarray,
) -> None:
    """Compare GRU-v2 residuals on validation and test data."""

    validation_residual = (
        validation_actual
        - validation_predicted
    ).ravel()

    test_residual = (
        test_actual
        - test_predicted
    ).ravel()

    rng = np.random.default_rng(42)

    def sample(values: np.ndarray) -> np.ndarray:
        if len(values) <= 200_000:
            return values

        indices = rng.choice(
            len(values),
            size=200_000,
            replace=False,
        )

        return values[indices]

    plot_df = pd.DataFrame(
        {
            "Validation": sample(
                validation_residual
            ),
            "Test": sample(
                test_residual
            ),
        }
    )

    plot_df = plot_df.melt(
        var_name="split",
        value_name="residual",
    )

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    sns.boxplot(
        data=plot_df,
        x="split",
        y="residual",
        ax=ax,
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_title(
        "GRU-v2 Residual Comparison"
    )

    ax.set_xlabel(
        "Dataset split"
    )

    ax.set_ylabel(
        "Residual"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "gru_residual_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Sprint 3.4D forecasting evaluation."""

    ensure_directories()

    print("=" * 64)
    print("Sprint 3.4D — Forecasting Model Evaluation")
    print("=" * 64)

    # -----------------------------------------------------------------------
    # Load actual targets
    # -----------------------------------------------------------------------

    print("\nLoading target arrays...")

    y_validation = load_array(
        "y_validation.npy"
    )

    y_test = load_array(
        "y_test.npy"
    )

    # -----------------------------------------------------------------------
    # Load saved GRU-v1/v2 predictions
    # -----------------------------------------------------------------------

    gru_v1_validation = np.load(
        PREDICTION_DIR
        / "gru_validation.npy"
    ).astype(np.float32)

    gru_v1_test = np.load(
        PREDICTION_DIR
        / "gru_test.npy"
    ).astype(np.float32)

    gru_v2_validation = np.load(
        PREDICTION_DIR
        / "gru_residual_validation.npy"
    ).astype(np.float32)

    gru_v2_test = np.load(
        PREDICTION_DIR
        / "gru_residual_test.npy"
    ).astype(np.float32)

    # -----------------------------------------------------------------------
    # Independently verify saved GRU metrics
    # -----------------------------------------------------------------------

    calculated_gru_v1_validation = evaluate_gru_v2(
        y_validation,
        gru_v1_validation,
        "Validation",
    )

    calculated_gru_v1_validation[
        "model_name"
    ] = "GRU-v1"

    calculated_gru_v1_test = evaluate_gru_v2(
        y_test,
        gru_v1_test,
        "Test",
    )

    calculated_gru_v1_test[
        "model_name"
    ] = "GRU-v1"

    calculated_gru_v2_validation = evaluate_gru_v2(
        y_validation,
        gru_v2_validation,
        "Validation",
    )

    calculated_gru_v2_test = evaluate_gru_v2(
        y_test,
        gru_v2_test,
        "Test",
    )

    calculated_results = pd.DataFrame(
        [
            calculated_gru_v1_validation,
            calculated_gru_v1_test,
            calculated_gru_v2_validation,
            calculated_gru_v2_test,
        ]
    )

    # -----------------------------------------------------------------------
    # Save recorded model comparison
    # -----------------------------------------------------------------------

    comparison = pd.DataFrame(
        BASELINE_RESULTS
    )

    comparison.to_csv(
        FORECASTING_DIR
        / "model_comparison.csv",
        index=False,
    )

    # -----------------------------------------------------------------------
    # Print comparison
    # -----------------------------------------------------------------------

    print("\nRecorded experiment results")
    print("=" * 64)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print("\nIndependent verification of GRU predictions")
    print("=" * 64)

    print(
        calculated_results.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    # -----------------------------------------------------------------------
    # Generate figures
    # -----------------------------------------------------------------------

    print("\nGenerating figures...")

    save_model_comparison(
        comparison
    )

    save_gru_actual_vs_predicted(
        y_test,
        gru_v2_test,
    )

    save_residual_distribution(
        y_test,
        gru_v2_test,
    )

    sensor_results = save_per_sensor_mae(
        y_test,
        gru_v2_test,
    )

    save_error_distribution_by_split(
        y_validation,
        gru_v2_validation,
        y_test,
        gru_v2_test,
    )

    # -----------------------------------------------------------------------
    # Save summary
    # -----------------------------------------------------------------------

    summary = {
        "experiment": "sprint_3.4d",
        "forecast_horizon_minutes": 5,
        "num_sensors": 207,
        "sequence_length": 12,
        "models": [
            "Persistence",
            "XGBoost",
            "GRU-v1",
            "GRU-v2",
        ],
        "figures": [
            "forecasting_model_comparison.png",
            "gru_actual_vs_predicted.png",
            "gru_residual_distribution.png",
            "gru_per_sensor_mae.png",
            "gru_residual_comparison.png",
        ],
        "prediction_level_evaluation": {
            "gru_v1_validation": (
                calculated_gru_v1_validation
            ),
            "gru_v1_test": (
                calculated_gru_v1_test
            ),
            "gru_v2_validation": (
                calculated_gru_v2_validation
            ),
            "gru_v2_test": (
                calculated_gru_v2_test
            ),
        },
        "sensor_count": int(
            len(sensor_results)
        ),
    }

    with (
        FORECASTING_DIR
        / "evaluation_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\nArtifacts generated")
    print("===================")
    print(
        FORECASTING_DIR
        / "model_comparison.csv"
    )
    print(
        FORECASTING_DIR
        / "per_sensor_mae.csv"
    )
    print(
        FORECASTING_DIR
        / "evaluation_summary.json"
    )

    print("\nFigures generated")
    print("=================")

    for figure in sorted(
        FIGURES_DIR.glob(
            "*.png"
        )
    ):
        print(figure)

    print("\nSprint 3.4D completed.")


if __name__ == "__main__":
    main()
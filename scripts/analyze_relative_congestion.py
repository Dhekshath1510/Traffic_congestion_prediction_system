from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ============================================================================
# Project paths
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

CLASSIFICATION_DIR = (
    RESULTS_DIR
    / "classification"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

CLASSIFICATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# Input files
# ============================================================================

TRAIN_TARGET_PATH = (
    SEQUENCE_DIR / "y_train.npy"
)

VALIDATION_TARGET_PATH = (
    SEQUENCE_DIR / "y_validation.npy"
)

TEST_TARGET_PATH = (
    SEQUENCE_DIR / "y_test.npy"
)


# ============================================================================
# Configuration
# ============================================================================

# Reference speed percentile.
#
# P85 is used as a robust approximation of high/free-flow operating speed.
# It is calculated ONLY from the training split.
REFERENCE_PERCENTILE = 85.0

# Speeds at or below this value are retained as observations but separately
# reported because zero/near-zero values may represent either true
# standstill traffic or sensor/data-state conditions.
NEAR_ZERO_SPEED = 1.0

# Candidate congestion-index thresholds.
#
# Congestion index:
#
#     C = 1 - (speed / reference_speed)
#
# Therefore:
#
# C = 0.00 -> operating near reference speed
# C = 0.20 -> approximately 20% reduction
# C = 0.40 -> approximately 40% reduction
#
# These are CANDIDATE thresholds and are not automatically the final rule.
CANDIDATE_THRESHOLDS = [
    (0.10, 0.30),
    (0.15, 0.35),
    (0.20, 0.40),
    (0.25, 0.45),
    (0.30, 0.50),
]


# ============================================================================
# Data loading
# ============================================================================


def load_target(path: Path) -> np.ndarray:
    """Load and validate a traffic-speed target array."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required target file does not exist: {path}"
        )

    values = np.load(path)

    if values.ndim != 2:
        raise ValueError(
            "Expected a 2-D array with shape "
            f"(samples, sensors), got {values.shape}"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            f"Target contains NaN or infinite values: {path}"
        )

    if np.any(values < 0):
        raise ValueError(
            f"Target contains negative traffic speeds: {path}"
        )

    return values.astype(
        np.float32,
        copy=False,
    )


# ============================================================================
# Reference-speed calculation
# ============================================================================


def calculate_reference_speeds(
    train_target: np.ndarray,
) -> np.ndarray:
    """
    Calculate sensor-specific reference speeds.

    The reference is calculated independently for every sensor using
    training data only.

    Near-zero observations are excluded from this calculation because
    they should not define the high/free-flow operating speed.
    """

    valid_training = np.where(
        train_target > NEAR_ZERO_SPEED,
        train_target,
        np.nan,
    )

    reference_speeds = np.nanpercentile(
        valid_training,
        REFERENCE_PERCENTILE,
        axis=0,
    )

    if not np.isfinite(reference_speeds).all():
        raise ValueError(
            "Unable to calculate a valid reference speed for "
            "one or more sensors."
        )

    if np.any(reference_speeds <= 0):
        raise ValueError(
            "One or more sensors have a non-positive reference speed."
        )

    return reference_speeds.astype(
        np.float32,
        copy=False,
    )


# ============================================================================
# Congestion index
# ============================================================================


def calculate_congestion_index(
    speeds: np.ndarray,
    reference_speeds: np.ndarray,
) -> np.ndarray:
    """
    Calculate sensor-relative congestion index.

        C = 1 - speed / reference_speed

    Values are clipped to [0, 1] for the congestion classification stage.

    A negative raw index means the observed speed is above the reference
    percentile. Such observations are treated as zero congestion.
    """

    if speeds.ndim != 2:
        raise ValueError(
            f"Expected 2-D speed array, got {speeds.shape}"
        )

    if reference_speeds.ndim != 1:
        raise ValueError(
            "Reference speeds must be a 1-D sensor vector."
        )

    if speeds.shape[1] != len(reference_speeds):
        raise ValueError(
            "Number of sensors differs between speed data and "
            f"reference speeds: {speeds.shape[1]} != "
            f"{len(reference_speeds)}"
        )

    raw_index = (
        1.0
        - (
            speeds
            / reference_speeds[np.newaxis, :]
        )
    )

    return np.clip(
        raw_index,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )


# ============================================================================
# Class assignment
# ============================================================================


def classify_congestion(
    congestion_index: np.ndarray,
    low_moderate_threshold: float,
    moderate_severe_threshold: float,
) -> np.ndarray:
    """
    Assign LOW, MODERATE, or SEVERE labels.

    Smaller congestion index means traffic is closer to normal/reference
    operation.

        C < low/moderate       -> LOW
        low/moderate <= C < moderate/severe -> MODERATE
        C >= moderate/severe   -> SEVERE
    """

    if (
        low_moderate_threshold
        >= moderate_severe_threshold
    ):
        raise ValueError(
            "LOW/MODERATE threshold must be smaller than "
            "MODERATE/SEVERE threshold."
        )

    return np.select(
        [
            congestion_index
            < low_moderate_threshold,

            congestion_index
            < moderate_severe_threshold,
        ],
        [
            "LOW",
            "MODERATE",
        ],
        default="SEVERE",
    )


# ============================================================================
# Candidate threshold analysis
# ============================================================================


def evaluate_thresholds(
    congestion_index: np.ndarray,
    split_name: str,
) -> pd.DataFrame:
    """
    Evaluate candidate thresholds for one dataset split.

    The returned percentages are calculated over every
    sample-sensor observation.
    """

    flattened = congestion_index.reshape(-1)

    rows: list[dict[str, float | str]] = []

    for (
        low_moderate,
        moderate_severe,
    ) in CANDIDATE_THRESHOLDS:

        labels = classify_congestion(
            flattened,
            low_moderate,
            moderate_severe,
        )

        counts = pd.Series(
            labels
        ).value_counts()

        total = len(labels)

        rows.append(
            {
                "split": split_name,
                "low_moderate_threshold": (
                    low_moderate
                ),
                "moderate_severe_threshold": (
                    moderate_severe
                ),
                "low_count": int(
                    counts.get("LOW", 0)
                ),
                "moderate_count": int(
                    counts.get("MODERATE", 0)
                ),
                "severe_count": int(
                    counts.get("SEVERE", 0)
                ),
                "low_percent": (
                    100.0
                    * counts.get("LOW", 0)
                    / total
                ),
                "moderate_percent": (
                    100.0
                    * counts.get("MODERATE", 0)
                    / total
                ),
                "severe_percent": (
                    100.0
                    * counts.get("SEVERE", 0)
                    / total
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# Distribution statistics
# ============================================================================


def describe_congestion_index(
    congestion_index: np.ndarray,
    split_name: str,
) -> dict[str, float | str]:
    """Generate descriptive statistics for congestion index."""

    values = congestion_index.reshape(-1)

    return {
        "split": split_name,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


# ============================================================================
# Sensor reference-speed statistics
# ============================================================================


def create_sensor_reference_dataframe(
    reference_speeds: np.ndarray,
) -> pd.DataFrame:
    """Create sensor-level reference-speed table."""

    sensor_ids = np.arange(
        1,
        len(reference_speeds) + 1,
    )

    return pd.DataFrame(
        {
            "sensor_index": sensor_ids,
            "reference_speed": reference_speeds,
        }
    )


# ============================================================================
# Visualizations
# ============================================================================


def plot_reference_speed_distribution(
    reference_speeds: np.ndarray,
) -> None:
    """Plot the distribution of sensor-specific reference speeds."""

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.hist(
        reference_speeds,
        bins=30,
    )

    ax.set_title(
        "Distribution of Sensor-Specific Reference Speeds"
    )

    ax.set_xlabel(
        "Reference speed"
    )

    ax.set_ylabel(
        "Number of sensors"
    )

    fig.tight_layout()

    output_path = (
        FIGURES_DIR
        / "sensor_reference_speed_distribution.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


def plot_congestion_index_distribution(
    congestion_indices: dict[str, np.ndarray],
) -> None:
    """Plot congestion-index distributions."""

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for split_name, values in congestion_indices.items():

        flattened = values.reshape(-1)

        # Downsample only for visualization.
        if len(flattened) > 150_000:
            rng = np.random.default_rng(42)

            indices = rng.choice(
                len(flattened),
                size=150_000,
                replace=False,
            )

            flattened = flattened[indices]

        ax.hist(
            flattened,
            bins=50,
            alpha=0.35,
            label=split_name,
        )

    ax.set_title(
        "Sensor-Relative Congestion Index Distribution"
    )

    ax.set_xlabel(
        "Congestion index"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        FIGURES_DIR
        / "relative_congestion_index_distribution.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


def plot_candidate_class_balance(
    candidate_results: pd.DataFrame,
) -> None:
    """Plot candidate congestion class distributions."""

    test_results = candidate_results.loc[
        candidate_results["split"] == "Test"
    ].copy()

    labels = [
        (
            f"{row.low_moderate_threshold:.2f}/"
            f"{row.moderate_severe_threshold:.2f}"
        )
        for row in test_results.itertuples()
    ]

    x = np.arange(
        len(test_results)
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        x - width,
        test_results["low_percent"],
        width,
        label="LOW",
    )

    ax.bar(
        x,
        test_results["moderate_percent"],
        width,
        label="MODERATE",
    )

    ax.bar(
        x + width,
        test_results["severe_percent"],
        width,
        label="SEVERE",
    )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_xlabel(
        "Candidate congestion-index thresholds"
    )

    ax.set_ylabel(
        "Samples (%)"
    )

    ax.set_title(
        "Candidate Relative Congestion Class Distribution"
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        FIGURES_DIR
        / "relative_congestion_class_balance.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


def plot_sensor_reference_speeds(
    reference_speeds: np.ndarray,
) -> None:
    """Plot reference speed for every sensor."""

    sensor_indices = np.arange(
        1,
        len(reference_speeds) + 1,
    )

    fig, ax = plt.subplots(
        figsize=(11, 4.5)
    )

    ax.plot(
        sensor_indices,
        reference_speeds,
        linewidth=1.0,
    )

    ax.set_title(
        "Sensor-Specific Reference Speed Across the Traffic Network"
    )

    ax.set_xlabel(
        "Sensor index"
    )

    ax.set_ylabel(
        "Reference speed"
    )

    fig.tight_layout()

    output_path = (
        FIGURES_DIR
        / "sensor_reference_speeds.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run Sprint 4.2 sensor-relative congestion analysis."""

    print("=" * 64)
    print("Sprint 4.2 — Sensor-Relative Congestion Analysis")
    print("=" * 64)

    # ------------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------------

    print("\nLoading target arrays...")

    train = load_target(
        TRAIN_TARGET_PATH
    )

    validation = load_target(
        VALIDATION_TARGET_PATH
    )

    test = load_target(
        TEST_TARGET_PATH
    )

    print("\nDataset shapes")
    print("==============")

    print(
        f"Train:       {train.shape}"
    )

    print(
        f"Validation:  {validation.shape}"
    )

    print(
        f"Test:        {test.shape}"
    )

    # ------------------------------------------------------------------------
    # Calculate reference speeds using TRAINING data only
    # ------------------------------------------------------------------------

    print("\nCalculating sensor reference speeds...")

    reference_speeds = calculate_reference_speeds(
        train
    )

    print(
        f"Reference percentile: "
        f"P{REFERENCE_PERCENTILE:.0f}"
    )

    print(
        f"Sensors: {len(reference_speeds)}"
    )

    print(
        f"Minimum reference speed: "
        f"{np.min(reference_speeds):.4f}"
    )

    print(
        f"Maximum reference speed: "
        f"{np.max(reference_speeds):.4f}"
    )

    print(
        f"Mean reference speed: "
        f"{np.mean(reference_speeds):.4f}"
    )

    print(
        f"Median reference speed: "
        f"{np.median(reference_speeds):.4f}"
    )

    # ------------------------------------------------------------------------
    # Save reference speeds
    # ------------------------------------------------------------------------

    reference_dataframe = (
        create_sensor_reference_dataframe(
            reference_speeds
        )
    )

    reference_path = (
        CLASSIFICATION_DIR
        / "sensor_reference_speeds.csv"
    )

    reference_dataframe.to_csv(
        reference_path,
        index=False,
    )

    print(
        f"\nSaved: {reference_path}"
    )

    # ------------------------------------------------------------------------
    # Calculate congestion indices
    # ------------------------------------------------------------------------

    print(
        "\nCalculating sensor-relative congestion indices..."
    )

    train_index = calculate_congestion_index(
        train,
        reference_speeds,
    )

    validation_index = calculate_congestion_index(
        validation,
        reference_speeds,
    )

    test_index = calculate_congestion_index(
        test,
        reference_speeds,
    )

    congestion_indices = {
        "Train": train_index,
        "Validation": validation_index,
        "Test": test_index,
    }

    # ------------------------------------------------------------------------
    # Distribution statistics
    # ------------------------------------------------------------------------

    distribution_rows = [
        describe_congestion_index(
            train_index,
            "Train",
        ),
        describe_congestion_index(
            validation_index,
            "Validation",
        ),
        describe_congestion_index(
            test_index,
            "Test",
        ),
    ]

    distribution_dataframe = pd.DataFrame(
        distribution_rows
    )

    distribution_path = (
        CLASSIFICATION_DIR
        / "relative_congestion_distribution.csv"
    )

    distribution_dataframe.to_csv(
        distribution_path,
        index=False,
    )

    print(
        f"\nSaved: {distribution_path}"
    )

    print(
        "\nCongestion-index statistics"
    )
    print(
        "==========================="
    )

    print(
        distribution_dataframe.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    # ------------------------------------------------------------------------
    # Candidate threshold analysis
    # ------------------------------------------------------------------------

    candidate_frames = [
        evaluate_thresholds(
            train_index,
            "Train",
        ),
        evaluate_thresholds(
            validation_index,
            "Validation",
        ),
        evaluate_thresholds(
            test_index,
            "Test",
        ),
    ]

    candidate_results = pd.concat(
        candidate_frames,
        ignore_index=True,
    )

    candidate_path = (
        CLASSIFICATION_DIR
        / "relative_congestion_thresholds.csv"
    )

    candidate_results.to_csv(
        candidate_path,
        index=False,
    )

    print(
        f"\nSaved: {candidate_path}"
    )

    print(
        "\nCandidate threshold analysis — Test"
    )
    print(
        "==================================="
    )

    print(
        candidate_results.loc[
            candidate_results["split"] == "Test"
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.2f}"
            ),
        )
    )

    # ------------------------------------------------------------------------
    # Near-zero observations
    # ------------------------------------------------------------------------

    print(
        "\nNear-zero speed observations"
    )
    print(
        "============================"
    )

    for split_name, values in (
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ):
        flattened = values.reshape(-1)

        count = np.count_nonzero(
            flattened <= NEAR_ZERO_SPEED
        )

        percentage = (
            100.0
            * count
            / len(flattened)
        )

        print(
            f"{split_name:<12} "
            f"{count:>10,} "
            f"({percentage:>6.2f}%)"
        )

    # ------------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------------

    print(
        "\nGenerating visualizations..."
    )

    plot_reference_speed_distribution(
        reference_speeds
    )

    plot_sensor_reference_speeds(
        reference_speeds
    )

    plot_congestion_index_distribution(
        congestion_indices
    )

    plot_candidate_class_balance(
        candidate_results
    )

    # ------------------------------------------------------------------------
    # Experiment metadata
    # ------------------------------------------------------------------------

    metadata = {
        "experiment": "sprint_4.2",
        "reference_speed_method": (
            "sensor-specific training-data P85"
        ),
        "reference_percentile": (
            REFERENCE_PERCENTILE
        ),
        "near_zero_speed_threshold": (
            NEAR_ZERO_SPEED
        ),
        "congestion_index": (
            "1 - observed_speed / reference_speed"
        ),
        "index_range": [
            0.0,
            1.0,
        ],
        "candidate_thresholds": [
            {
                "low_moderate": low,
                "moderate_severe": severe,
            }
            for low, severe
            in CANDIDATE_THRESHOLDS
        ],
        "sensor_count": int(
            train.shape[1]
        ),
    }

    metadata_path = (
        CLASSIFICATION_DIR
        / "relative_congestion_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"Saved: {metadata_path}"
    )

    print(
        "\n" + "=" * 64
    )

    print(
        "Sprint 4.2 completed."
    )

    print(
        "=" * 64
    )


if __name__ == "__main__":
    main()
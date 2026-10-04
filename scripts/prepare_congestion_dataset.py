from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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

CLASSIFICATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "congestion"
)

RESULTS_CLASSIFICATION_DIR = (
    PROJECT_ROOT
    / "results"
    / "classification"
)

RESULTS_FIGURES_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)


CLASSIFICATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_CLASSIFICATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# Input artifacts
# ============================================================================

REFERENCE_SPEEDS_PATH = (
    RESULTS_CLASSIFICATION_DIR
    / "sensor_reference_speeds.csv"
)


SPLITS = {
    "train": {
        "X": SEQUENCE_DIR / "X_train.npy",
        "y": SEQUENCE_DIR / "y_train.npy",
        "timestamps": SEQUENCE_DIR / "timestamps_train.npy",
    },
    "validation": {
        "X": SEQUENCE_DIR / "X_validation.npy",
        "y": SEQUENCE_DIR / "y_validation.npy",
        "timestamps": SEQUENCE_DIR / "timestamps_validation.npy",
    },
    "test": {
        "X": SEQUENCE_DIR / "X_test.npy",
        "y": SEQUENCE_DIR / "y_test.npy",
        "timestamps": SEQUENCE_DIR / "timestamps_test.npy",
    },
}


# ============================================================================
# Final congestion-label definition
# ============================================================================

REFERENCE_PERCENTILE = 85.0

LOW_MODERATE_THRESHOLD = 0.10
MODERATE_SEVERE_THRESHOLD = 0.30

CLASS_TO_ID = {
    "LOW": 0,
    "MODERATE": 1,
    "SEVERE": 2,
}

ID_TO_CLASS = {
    0: "LOW",
    1: "MODERATE",
    2: "SEVERE",
}


# ============================================================================
# Data loading
# ============================================================================


def load_array(
    path: Path,
    expected_ndim: int | None = None,
) -> np.ndarray:
    """Load a NumPy artifact and perform basic validation."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact does not exist: {path}"
        )

    array = np.load(
        path,
        allow_pickle=False,
    )

    if expected_ndim is not None and array.ndim != expected_ndim:
        raise ValueError(
            f"Expected {expected_ndim}-D array for {path.name}, "
            f"got shape {array.shape}"
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"Artifact contains NaN or infinite values: {path}"
        )

    return array


def load_reference_speeds() -> np.ndarray:
    """
    Load the fixed sensor-specific P85 reference speeds.

    These values were calculated during Sprint 4.2 from TRAINING data only.
    """

    if not REFERENCE_SPEEDS_PATH.exists():
        raise FileNotFoundError(
            "Sprint 4.2 reference-speed artifact was not found:\n"
            f"{REFERENCE_SPEEDS_PATH}"
        )

    dataframe = pd.read_csv(
        REFERENCE_SPEEDS_PATH
    )

    required_columns = {
        "sensor_index",
        "reference_speed",
    }

    missing = required_columns.difference(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            "Reference-speed file is missing columns: "
            f"{sorted(missing)}"
        )

    reference_speeds = dataframe[
        "reference_speed"
    ].to_numpy(
        dtype=np.float32
    )

    if len(reference_speeds) == 0:
        raise ValueError(
            "Reference-speed file contains no sensors."
        )

    if not np.isfinite(reference_speeds).all():
        raise ValueError(
            "Reference speeds contain invalid values."
        )

    if np.any(reference_speeds <= 0):
        raise ValueError(
            "Reference speeds must be positive."
        )

    return reference_speeds


# ============================================================================
# Feature engineering
# ============================================================================


def build_features(
    X: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    Build compact sample-level traffic features.

    Input:
        X -> (samples, sequence_length, sensors)

    Output:
        features -> (samples, engineered_features)

    Features intentionally summarize the temporal traffic sequence rather
    than flattening all 12 x 207 values. This keeps the classification
    dataset compact and suitable for tree-based models.
    """

    if X.ndim != 3:
        raise ValueError(
            f"Expected X with shape "
            "(samples, sequence_length, sensors), "
            f"got {X.shape}"
        )

    samples, sequence_length, sensors = X.shape

    if timestamps.shape[0] != samples:
        raise ValueError(
            "Timestamp count does not match X samples: "
            f"{timestamps.shape[0]} != {samples}"
        )

    current = X[:, -1, :]

    previous = X[:, -2, :]

    oldest = X[:, 0, :]

    sequence_mean = np.mean(
        X,
        axis=(1, 2),
    )

    sequence_std = np.std(
        X,
        axis=(1, 2),
    )

    current_mean = np.mean(
        current,
        axis=1,
    )

    current_std = np.std(
        current,
        axis=1,
    )

    current_min = np.min(
        current,
        axis=1,
    )

    current_max = np.max(
        current,
        axis=1,
    )

    previous_mean = np.mean(
        previous,
        axis=1,
    )

    previous_std = np.std(
        previous,
        axis=1,
    )

    speed_change_mean = (
        current_mean
        - previous_mean
    )

    speed_change_std = np.std(
        current - previous,
        axis=1,
    )

    long_term_change = (
        current_mean
        - np.mean(
            oldest,
            axis=1,
        )
    )

    # Recent temporal statistics.
    recent_window = X[:, -3:, :]

    recent_mean = np.mean(
        recent_window,
        axis=(1, 2),
    )

    recent_std = np.std(
        recent_window,
        axis=(1, 2),
    )

    # Slow-moving traffic fraction in the current observation.
    near_zero_fraction = np.mean(
        current <= 1.0,
        axis=1,
    )

    low_speed_fraction = np.mean(
        current < 30.0,
        axis=1,
    )

    moderate_speed_fraction = np.mean(
        (current >= 30.0)
        & (current < 50.0),
        axis=1,
    )

    high_speed_fraction = np.mean(
        current >= 60.0,
        axis=1,
    )

    # Temporal information.
    timestamp_series = pd.to_datetime(
        timestamps
    )

    hour = (
        timestamp_series.hour
        + timestamp_series.minute / 60.0
    ).to_numpy(
        dtype=np.float32
    )

    day_of_week = (
        timestamp_series.dayofweek
    ).to_numpy(
        dtype=np.float32
    )

    is_weekend = (
        day_of_week >= 5
    ).astype(
        np.float32
    )

    # Cyclic encoding prevents the model from treating 23:55 and 00:00
    # as maximally distant values.
    hour_angle = (
        2.0
        * np.pi
        * hour
        / 24.0
    )

    hour_sin = np.sin(
        hour_angle
    ).astype(
        np.float32
    )

    hour_cos = np.cos(
        hour_angle
    ).astype(
        np.float32
    )

    day_angle = (
        2.0
        * np.pi
        * day_of_week
        / 7.0
    )

    day_sin = np.sin(
        day_angle
    ).astype(
        np.float32
    )

    day_cos = np.cos(
        day_angle
    ).astype(
        np.float32
    )

    features = np.column_stack(
        [
            current_mean,
            current_std,
            current_min,
            current_max,
            previous_mean,
            previous_std,
            speed_change_mean,
            speed_change_std,
            long_term_change,
            sequence_mean,
            sequence_std,
            recent_mean,
            recent_std,
            near_zero_fraction,
            low_speed_fraction,
            moderate_speed_fraction,
            high_speed_fraction,
            hour,
            hour_sin,
            hour_cos,
            day_of_week,
            day_sin,
            day_cos,
            is_weekend,
        ]
    ).astype(
        np.float32
    )

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
    ]

    if features.shape != (
        samples,
        len(feature_names),
    ):
        raise RuntimeError(
            "Feature matrix shape does not match feature names: "
            f"{features.shape} vs "
            f"{len(feature_names)}"
        )

    return features, feature_names


# ============================================================================
# Congestion labels
# ============================================================================


def calculate_congestion_index(
    target: np.ndarray,
    reference_speeds: np.ndarray,
) -> np.ndarray:
    """
    Calculate future sensor-relative congestion index.

        CI = 1 - future_speed / reference_speed

    The reference speed is fixed from Sprint 4.2.
    """

    if target.ndim != 2:
        raise ValueError(
            f"Expected target shape (samples, sensors), "
            f"got {target.shape}"
        )

    if target.shape[1] != len(reference_speeds):
        raise ValueError(
            "Target sensor count does not match reference-speed count: "
            f"{target.shape[1]} != {len(reference_speeds)}"
        )

    raw_index = (
        1.0
        - (
            target
            / reference_speeds[
                np.newaxis,
                :
            ]
        )
    )

    return np.clip(
        raw_index,
        0.0,
        1.0,
    ).astype(
        np.float32
    )


def create_labels(
    congestion_index: np.ndarray,
) -> np.ndarray:
    """
    Convert congestion index into integer classes.

        0 = LOW
        1 = MODERATE
        2 = SEVERE
    """

    return np.select(
        [
            congestion_index
            < LOW_MODERATE_THRESHOLD,

            congestion_index
            < MODERATE_SEVERE_THRESHOLD,
        ],
        [
            CLASS_TO_ID["LOW"],
            CLASS_TO_ID["MODERATE"],
        ],
        default=CLASS_TO_ID["SEVERE"],
    ).astype(
        np.int8
    )


# ============================================================================
# Statistics
# ============================================================================


def calculate_class_statistics(
    labels: np.ndarray,
    split_name: str,
) -> pd.DataFrame:
    """Calculate class counts and percentages."""

    flattened = labels.reshape(-1)

    total = len(flattened)

    rows = []

    for class_id, class_name in ID_TO_CLASS.items():
        count = int(
            np.count_nonzero(
                flattened == class_id
            )
        )

        rows.append(
            {
                "split": split_name,
                "class_id": class_id,
                "class_name": class_name,
                "count": count,
                "percentage": (
                    100.0
                    * count
                    / total
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_sensor_statistics(
    labels: np.ndarray,
) -> pd.DataFrame:
    """Calculate class distribution independently for every sensor."""

    rows = []

    sensors = labels.shape[1]

    for sensor_index in range(sensors):
        sensor_labels = labels[
            :,
            sensor_index,
        ]

        total = len(sensor_labels)

        rows.append(
            {
                "sensor_index": sensor_index + 1,
                "low_percent": (
                    100.0
                    * np.mean(
                        sensor_labels == 0
                    )
                ),
                "moderate_percent": (
                    100.0
                    * np.mean(
                        sensor_labels == 1
                    )
                ),
                "severe_percent": (
                    100.0
                    * np.mean(
                        sensor_labels == 2
                    )
                ),
                "sample_count": total,
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# Save one dataset split
# ============================================================================


def process_split(
    split_name: str,
    paths: dict[str, Path],
    reference_speeds: np.ndarray,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Process one dataset split."""

    print(
        f"\nProcessing {split_name}..."
    )

    X = load_array(
        paths["X"],
        expected_ndim=3,
    )

    y = load_array(
        paths["y"],
        expected_ndim=2,
    )

    timestamps = load_array(
        paths["timestamps"]
    )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{split_name}: X/y sample mismatch: "
            f"{X.shape[0]} != {y.shape[0]}"
        )

    if X.shape[2] != y.shape[1]:
        raise ValueError(
            f"{split_name}: X/y sensor mismatch: "
            f"{X.shape[2]} != {y.shape[1]}"
        )

    print(
        f"X shape: {X.shape}"
    )

    print(
        f"Target shape: {y.shape}"
    )

    # ------------------------------------------------------------------------
    # Feature generation
    # ------------------------------------------------------------------------

    features, feature_names = build_features(
        X,
        timestamps,
    )

    # ------------------------------------------------------------------------
    # Future congestion index
    # ------------------------------------------------------------------------

    congestion_index = (
        calculate_congestion_index(
            y,
            reference_speeds,
        )
    )

    labels = create_labels(
        congestion_index
    )

    # ------------------------------------------------------------------------
    # Save compact arrays
    # ------------------------------------------------------------------------

    np.save(
        CLASSIFICATION_DIR
        / f"X_{split_name}.npy",
        features,
    )

    np.save(
        CLASSIFICATION_DIR
        / f"y_{split_name}.npy",
        labels,
    )

    np.save(
        CLASSIFICATION_DIR
        / f"congestion_index_{split_name}.npy",
        congestion_index,
    )

    np.save(
        CLASSIFICATION_DIR
        / f"timestamps_{split_name}.npy",
        timestamps,
    )

    # ------------------------------------------------------------------------
    # Save human-readable feature table
    # ------------------------------------------------------------------------

    feature_dataframe = pd.DataFrame(
        features,
        columns=feature_names,
    )

    feature_dataframe.insert(
        0,
        "timestamp",
        pd.to_datetime(
            timestamps
        ),
    )

    feature_dataframe.to_csv(
        CLASSIFICATION_DIR
        / f"features_{split_name}.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Class statistics
    # ------------------------------------------------------------------------

    class_statistics = calculate_class_statistics(
        labels,
        split_name,
    )

    sensor_statistics = calculate_sensor_statistics(
        labels
    )

    sensor_statistics.insert(
        0,
        "split",
        split_name,
    )

    print(
        "\nClass distribution:"
    )

    print(
        class_statistics.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.2f}"
            ),
        )
    )

    print(
        f"Saved X_{split_name}.npy: "
        f"{features.shape}"
    )

    print(
        f"Saved y_{split_name}.npy: "
        f"{labels.shape}"
    )

    return (
        class_statistics,
        sensor_statistics,
    )


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run Sprint 4.3."""

    print("=" * 64)
    print("Sprint 4.3 — Congestion Dataset & Label Generation")
    print("=" * 64)

    print(
        "\nFinal congestion-label definition"
    )

    print(
        "================================="
    )

    print(
        f"Reference speed: training-data P"
        f"{REFERENCE_PERCENTILE:.0f}"
    )

    print(
        f"LOW:       CI < "
        f"{LOW_MODERATE_THRESHOLD:.2f}"
    )

    print(
        f"MODERATE:  "
        f"{LOW_MODERATE_THRESHOLD:.2f} <= CI < "
        f"{MODERATE_SEVERE_THRESHOLD:.2f}"
    )

    print(
        f"SEVERE:    CI >= "
        f"{MODERATE_SEVERE_THRESHOLD:.2f}"
    )

    # ------------------------------------------------------------------------
    # Load fixed reference speeds
    # ------------------------------------------------------------------------

    print(
        "\nLoading Sprint 4.2 reference speeds..."
    )

    reference_speeds = (
        load_reference_speeds()
    )

    print(
        f"Sensors: {len(reference_speeds)}"
    )

    # ------------------------------------------------------------------------
    # Process all splits
    # ------------------------------------------------------------------------

    class_statistics = []
    sensor_statistics = []

    for split_name, paths in SPLITS.items():

        split_class_statistics, split_sensor_statistics = (
            process_split(
                split_name,
                paths,
                reference_speeds,
            )
        )

        class_statistics.append(
            split_class_statistics
        )

        sensor_statistics.append(
            split_sensor_statistics
        )

    # ------------------------------------------------------------------------
    # Combine statistics
    # ------------------------------------------------------------------------

    class_distribution = pd.concat(
        class_statistics,
        ignore_index=True,
    )

    sensor_distribution = pd.concat(
        sensor_statistics,
        ignore_index=True,
    )

    class_distribution_path = (
        RESULTS_CLASSIFICATION_DIR
        / "congestion_class_distribution.csv"
    )

    sensor_distribution_path = (
        RESULTS_CLASSIFICATION_DIR
        / "sensor_congestion_distribution.csv"
    )

    class_distribution.to_csv(
        class_distribution_path,
        index=False,
    )

    sensor_distribution.to_csv(
        sensor_distribution_path,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------------------

    metadata = {
        "experiment": "sprint_4.3",
        "label_definition": {
            "reference_speed": (
                "sensor-specific P85 calculated "
                "from training data only"
            ),
            "congestion_index": (
                "1 - future_speed / reference_speed"
            ),
            "low": (
                f"CI < {LOW_MODERATE_THRESHOLD:.2f}"
            ),
            "moderate": (
                f"{LOW_MODERATE_THRESHOLD:.2f} <= CI < "
                f"{MODERATE_SEVERE_THRESHOLD:.2f}"
            ),
            "severe": (
                f"CI >= {MODERATE_SEVERE_THRESHOLD:.2f}"
            ),
        },
        "class_mapping": CLASS_TO_ID,
        "num_sensors": int(
            len(reference_speeds)
        ),
        "feature_count": 24,
        "feature_design": (
            "Compact temporal, network traffic, "
            "speed-change, and calendar features"
        ),
        "forecast_target": (
            "Future traffic speed at the next "
            "5-minute prediction horizon"
        ),
        "data_leakage_control": (
            "Reference speeds are calculated from "
            "training data only and reused unchanged "
            "for validation and test."
        ),
    }

    metadata_path = (
        RESULTS_CLASSIFICATION_DIR
        / "congestion_label_metadata.json"
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

    # ------------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------------

    print(
        "\nGenerated artifacts"
    )

    print(
        "===================="
    )

    print(
        f"Data directory: "
        f"{CLASSIFICATION_DIR}"
    )

    print(
        f"Class statistics: "
        f"{class_distribution_path}"
    )

    print(
        f"Sensor statistics: "
        f"{sensor_distribution_path}"
    )

    print(
        f"Metadata: "
        f"{metadata_path}"
    )

    print(
        "\nFinal dataset shapes:"
    )

    for split_name in SPLITS:
        X_path = (
            CLASSIFICATION_DIR
            / f"X_{split_name}.npy"
        )

        y_path = (
            CLASSIFICATION_DIR
            / f"y_{split_name}.npy"
        )

        X = np.load(X_path)
        y = np.load(y_path)

        print(
            f"{split_name:<12} "
            f"X={X.shape} "
            f"y={y.shape}"
        )

    print(
        "\n" + "=" * 64
    )

    print(
        "Sprint 4.3 completed."
    )

    print(
        "=" * 64
    )


if __name__ == "__main__":
    main()
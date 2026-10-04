from __future__ import annotations
from datetime import datetime
import numpy as np

EXPECTED_SENSOR_COUNT = 207
MINIMUM_SEQUENCE_LENGTH = 2
RECENT_WINDOW_LENGTH = 3


BASE_FEATURE_NAMES: tuple[str, ...] = (
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
)

GRU_FEATURE_NAME = "gru_predicted_speed"
SENSOR_FEATURE_NAME = "sensor_index"

CLASSIFIER_FEATURE_NAMES: tuple[str, ...] = (
    *BASE_FEATURE_NAMES,
    GRU_FEATURE_NAME,
    SENSOR_FEATURE_NAME,
)


def build_base_features(
    sequence: np.ndarray,
    timestamp: datetime,
) -> np.ndarray:
    """Build the 24 historical features used by the classifier."""

    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    if sequence.ndim != 2:
        raise ValueError(
            "Expected sequence with shape "
            "(timesteps, sensors), "
            f"got {sequence.shape}."
        )

    timesteps, sensors = sequence.shape

    if timesteps < MINIMUM_SEQUENCE_LENGTH:
        raise ValueError(
            "Sequence must contain at least "
            f"{MINIMUM_SEQUENCE_LENGTH} timesteps, "
            f"got {timesteps}."
        )

    if sensors != EXPECTED_SENSOR_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SENSOR_COUNT} sensors, "
            f"got {sensors}."
        )

    if not np.isfinite(sequence).all():
        raise ValueError(
            "Sequence contains non-finite values."
        )

    current = sequence[-1, :]
    previous = sequence[-2, :]
    oldest = sequence[0, :]

    sequence_mean = np.mean(sequence)
    sequence_std = np.std(sequence)

    current_mean = np.mean(current)
    current_std = np.std(current)
    current_min = np.min(current)
    current_max = np.max(current)

    previous_mean = np.mean(previous)
    previous_std = np.std(previous)

    speed_change = current - previous

    speed_change_mean = (
        current_mean - previous_mean
    )

    speed_change_std = np.std(
        speed_change
    )

    long_term_change = (
        current_mean - np.mean(oldest)
    )

    recent_window = sequence[
        -RECENT_WINDOW_LENGTH:
    ]

    recent_mean = np.mean(
        recent_window
    )

    recent_std = np.std(
        recent_window
    )

    near_zero_fraction = np.mean(
        current <= 1.0
    )

    low_speed_fraction = np.mean(
        current < 30.0
    )

    moderate_speed_fraction = np.mean(
        (current >= 30.0)
        & (current < 50.0)
    )

    high_speed_fraction = np.mean(
        current >= 60.0
    )

    hour = np.float32(
        timestamp.hour
        + timestamp.minute / 60.0
    )

    day_of_week = np.float32(
        timestamp.weekday()
    )

    is_weekend = np.float32(
        day_of_week >= 5
    )

    hour_angle = (
        2.0 * np.pi * hour / 24.0
    )

    hour_sin = np.float32(
        np.sin(hour_angle)
    )

    hour_cos = np.float32(
        np.cos(hour_angle)
    )

    day_angle = (
        2.0 * np.pi * day_of_week / 7.0
    )

    day_sin = np.float32(
        np.sin(day_angle)
    )

    day_cos = np.float32(
        np.cos(day_angle)
    )

    features = np.array(
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
        ],
        dtype=np.float32,
    )

    if features.shape != (24,):
        raise RuntimeError(
            f"Base feature shape mismatch: {features.shape}"
        )

    return features


def build_classifier_features(
    sequence: np.ndarray,
    timestamp: datetime,
    gru_predicted_speed: np.ndarray,
) -> np.ndarray:
    """
    Build XGBoost features using the GRU future-speed prediction.

    Output shape:
        (207, 26)

    Features:
        24 historical/context features
        1 GRU predicted future speed
        1 sensor index
    """

    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    gru_predicted_speed = np.asarray(
        gru_predicted_speed,
        dtype=np.float32,
    )

    if gru_predicted_speed.shape != (
        EXPECTED_SENSOR_COUNT,
    ):
        raise ValueError(
            "Expected GRU prediction shape "
            f"({EXPECTED_SENSOR_COUNT},), "
            f"got {gru_predicted_speed.shape}."
        )

    if not np.isfinite(
        gru_predicted_speed
    ).all():
        raise ValueError(
            "GRU predicted speeds contain "
            "NaN or infinite values."
        )

    base_features = build_base_features(
        sequence=sequence,
        timestamp=timestamp,
    )

    repeated_features = np.repeat(
        base_features.reshape(1, -1),
        EXPECTED_SENSOR_COUNT,
        axis=0,
    )

    predicted_speed_column = (
        gru_predicted_speed.reshape(-1, 1)
    )

    sensor_indices = np.arange(
        EXPECTED_SENSOR_COUNT,
        dtype=np.float32,
    ).reshape(-1, 1)

    classifier_features = np.column_stack(
        [
            repeated_features,
            predicted_speed_column,
            sensor_indices,
        ]
    ).astype(
        np.float32,
        copy=False,
    )

    expected_shape = (
        EXPECTED_SENSOR_COUNT,
        26,
    )

    if classifier_features.shape != expected_shape:
        raise RuntimeError(
            "Classifier feature shape mismatch: "
            f"{classifier_features.shape}; "
            f"expected {expected_shape}."
        )

    return classifier_features
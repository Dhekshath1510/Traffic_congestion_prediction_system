from __future__ import annotations

import numpy as np


def build_temporal_features(
    features: np.ndarray,
) -> np.ndarray:
    """Build compact temporal features from traffic history.

    For every sensor, the following features are generated:

    - latest observed speed
    - historical mean speed
    - historical standard deviation
    - linear trend across the history window

    Args:
        features:
            Traffic history with shape:
            (samples, history_steps, sensors).

    Returns:
        Feature matrix with shape:
        (samples, sensors * 4).
    """
    values = np.asarray(
        features,
        dtype=np.float32,
    )

    if values.ndim != 3:
        raise ValueError(
            "Expected features with shape "
            "(samples, history_steps, sensors). "
            f"Received: {values.shape}",
        )

    if values.shape[1] < 2:
        raise ValueError(
            "At least two history steps are required " "to calculate temporal trends.",
        )

    latest = values[:, -1, :]

    mean = np.mean(
        values,
        axis=1,
    )

    standard_deviation = np.std(
        values,
        axis=1,
    )

    trend = _calculate_trend(
        values,
    )

    return np.concatenate(
        [
            latest,
            mean,
            standard_deviation,
            trend,
        ],
        axis=1,
    )


def _calculate_trend(
    values: np.ndarray,
) -> np.ndarray:
    """Calculate a linear speed trend for every sensor."""
    history_steps = values.shape[1]

    time_index = np.arange(
        history_steps,
        dtype=np.float32,
    )

    centered_time = time_index - np.mean(time_index)

    denominator = np.sum(
        centered_time**2,
    )

    centered_values = values - np.mean(
        values,
        axis=1,
        keepdims=True,
    )

    numerator = np.sum(
        centered_values * centered_time.reshape(1, -1, 1),
        axis=1,
    )

    return numerator / denominator

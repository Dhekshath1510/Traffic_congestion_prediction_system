import numpy as np


def mean_absolute_error(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate mean absolute error across all samples and sensors."""
    actual_array = np.asarray(actual, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)

    if actual_array.shape != predicted_array.shape:
        raise ValueError(
            "Actual and predicted arrays must have the same shape.",
        )

    return float(
        np.mean(
            np.abs(
                actual_array - predicted_array,
            ),
        ),
    )


def root_mean_squared_error(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate root mean squared error across all samples and sensors."""
    actual_array = np.asarray(actual, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)

    if actual_array.shape != predicted_array.shape:
        raise ValueError(
            "Actual and predicted arrays must have the same shape.",
        )

    return float(
        np.sqrt(
            np.mean(
                (actual_array - predicted_array) ** 2,
            ),
        ),
    )


def symmetric_mean_absolute_percentage_error(
    actual: np.ndarray,
    predicted: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Calculate symmetric mean absolute percentage error.

    Entries where both actual and predicted values are effectively
    zero are excluded because their denominator is zero.
    """
    actual_array = np.asarray(actual, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)

    if actual_array.shape != predicted_array.shape:
        raise ValueError(
            "Actual and predicted arrays must have the same shape.",
        )

    denominator = np.abs(actual_array) + np.abs(predicted_array)

    valid = denominator > epsilon

    if not np.any(valid):
        return 0.0

    numerator = np.abs(
        actual_array[valid] - predicted_array[valid],
    )

    return float(
        np.mean(
            2.0 * numerator / denominator[valid],
        )
        * 100.0,
    )


def sensor_mean_absolute_error(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    """Calculate MAE independently for each sensor.

    Args:
        actual:
            Array with shape (samples, sensors).

        predicted:
            Array with shape (samples, sensors).

    Returns:
        One MAE value for each sensor.
    """
    actual_array = np.asarray(actual, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)

    if actual_array.ndim != 2:
        raise ValueError(
            "Expected arrays with shape (samples, sensors).",
        )

    if actual_array.shape != predicted_array.shape:
        raise ValueError(
            "Actual and predicted arrays must have the same shape.",
        )

    return np.mean(
        np.abs(
            actual_array - predicted_array,
        ),
        axis=0,
    )

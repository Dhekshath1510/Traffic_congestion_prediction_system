from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceConfig:
    """Configuration for supervised traffic sequences."""

    history_steps: int = 12
    horizon_steps: int = 3


@dataclass(frozen=True)
class TrafficSequenceDataset:
    """Generated supervised traffic sequences."""

    features: np.ndarray
    targets: np.ndarray
    timestamps: np.ndarray


class TrafficSequenceGenerator:
    """Generate temporal sequences for traffic forecasting."""

    def __init__(
        self,
        config: SequenceConfig | None = None,
    ) -> None:
        """Initialize the sequence generator."""
        self._config = config or SequenceConfig()

    def generate(
        self,
        dataframe: pd.DataFrame,
    ) -> TrafficSequenceDataset:
        """Generate multi-sensor forecasting sequences.

        The latest historical observation is used to predict
        the traffic state at the configured future horizon.

        Args:
            dataframe: Time-indexed traffic dataframe.

        Returns:
            Features, multi-sensor targets and target timestamps.
        """
        if dataframe.empty:
            raise ValueError(
                "Cannot generate sequences from an empty dataset.",
            )

        if self._config.history_steps <= 0:
            raise ValueError(
                "history_steps must be greater than zero.",
            )

        if self._config.horizon_steps <= 0:
            raise ValueError(
                "horizon_steps must be greater than zero.",
            )

        timestamp_column = "timestamp"

        if timestamp_column not in dataframe.columns:
            raise ValueError(
                "Dataset must contain a timestamp column.",
            )

        ordered = dataframe.sort_values(
            timestamp_column,
        ).reset_index(drop=True)

        timestamps = pd.to_datetime(
            ordered[timestamp_column],
        )

        sensor_columns = [
            column for column in ordered.columns if column != timestamp_column
        ]

        if not sensor_columns:
            raise ValueError(
                "Dataset must contain at least one sensor column.",
            )

        sensor_values = ordered[sensor_columns].to_numpy(
            dtype=np.float32,
        )

        history_steps = self._config.history_steps
        horizon_steps = self._config.horizon_steps

        sample_count = len(sensor_values) - history_steps - horizon_steps + 1

        if sample_count <= 0:
            raise ValueError(
                "Dataset does not contain enough observations "
                "to generate the requested sequences.",
            )

        features = np.empty(
            (
                sample_count,
                history_steps,
                len(sensor_columns),
            ),
            dtype=np.float32,
        )

        targets = np.empty(
            (
                sample_count,
                len(sensor_columns),
            ),
            dtype=np.float32,
        )

        target_timestamps = np.empty(
            sample_count,
            dtype="datetime64[ns]",
        )

        for index in range(sample_count):
            history_end = index + history_steps

            target_index = history_end + horizon_steps - 1

            features[index] = sensor_values[index:history_end]

            targets[index] = sensor_values[target_index]

            target_timestamps[index] = timestamps.iloc[target_index]

        return TrafficSequenceDataset(
            features=features,
            targets=targets,
            timestamps=target_timestamps,
        )

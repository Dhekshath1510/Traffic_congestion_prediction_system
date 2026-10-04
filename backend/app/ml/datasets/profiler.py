from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class DatasetProfile:
    """Summary statistics for a traffic time-series dataset."""

    row_count: int
    sensor_count: int
    duplicate_rows: int
    start_timestamp: pd.Timestamp | None
    end_timestamp: pd.Timestamp | None
    expected_interval_minutes: int
    median_interval_minutes: float | None
    missing_timestamp_count: int
    missing_value_count: int
    invalid_value_count: int
    sensor_statistics: dict[str, float]


class TrafficDatasetProfiler:
    """Generate reproducible traffic dataset profiles."""

    def profile(
        self,
        dataframe: pd.DataFrame,
        expected_interval_minutes: int = 5,
    ) -> DatasetProfile:
        """Profile a wide traffic time-series dataframe.

        Args:
            dataframe: Traffic dataframe.
            expected_interval_minutes: Expected sampling interval.

        Returns:
            Dataset profile.
        """

        if "timestamp" not in dataframe.columns:
            raise ValueError(
                "Dataset must contain a timestamp column.",
            )

        timestamps = pd.to_datetime(
            dataframe["timestamp"],
            errors="coerce",
        )

        valid_timestamps = timestamps.dropna()

        start_timestamp = valid_timestamps.min() if not valid_timestamps.empty else None

        end_timestamp = valid_timestamps.max() if not valid_timestamps.empty else None

        intervals = (
            valid_timestamps.sort_values().diff().dropna().dt.total_seconds().div(60)
        )

        median_interval = float(intervals.median()) if not intervals.empty else None

        expected_delta = pd.Timedelta(
            minutes=expected_interval_minutes,
        )

        missing_timestamps = 0

        if not valid_timestamps.empty:
            unique_timestamps = pd.DatetimeIndex(
                valid_timestamps.drop_duplicates().sort_values(),
            )

            full_range = pd.date_range(
                start=unique_timestamps[0],
                end=unique_timestamps[-1],
                freq=expected_delta,
            )

            missing_timestamps = len(
                full_range.difference(unique_timestamps),
            )

        sensor_columns = [
            column for column in dataframe.columns if column != "timestamp"
        ]

        sensor_data = dataframe[sensor_columns]

        missing_value_count = int(
            sensor_data.isna().sum().sum(),
        )

        numeric_sensor_data = sensor_data.apply(
            pd.to_numeric,
            errors="coerce",
        )

        invalid_value_count = int(
            (numeric_sensor_data < 0).sum().sum(),
        )

        sensor_statistics: dict[str, float] = {}

        if not numeric_sensor_data.empty:
            sensor_statistics = {
                "minimum": float(
                    numeric_sensor_data.min().min(),
                ),
                "maximum": float(
                    numeric_sensor_data.max().max(),
                ),
                "mean": float(
                    numeric_sensor_data.mean().mean(),
                ),
                "median": float(
                    numeric_sensor_data.median().median(),
                ),
            }

        return DatasetProfile(
            row_count=len(dataframe),
            sensor_count=len(sensor_columns),
            duplicate_rows=int(
                dataframe.duplicated().sum(),
            ),
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            expected_interval_minutes=expected_interval_minutes,
            median_interval_minutes=median_interval,
            missing_timestamp_count=missing_timestamps,
            missing_value_count=missing_value_count,
            invalid_value_count=invalid_value_count,
            sensor_statistics=sensor_statistics,
        )

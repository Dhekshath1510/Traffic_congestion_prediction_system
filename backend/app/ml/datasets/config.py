from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a traffic time-series dataset."""

    timestamp_column: str = "timestamp"
    expected_interval_minutes: int = 5
    prediction_horizon_steps: int = 3
    prediction_horizon_minutes: int = 15

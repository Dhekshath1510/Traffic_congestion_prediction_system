from pathlib import Path

import pandas as pd


class TrafficDatasetLoader:
    """Load traffic time-series datasets."""

    def load_csv(self, path: Path) -> pd.DataFrame:
        """Load a traffic dataset from CSV.

        Args:
            path: Path to the CSV file.

        Returns:
            Normalized traffic time-series dataframe.

        Raises:
            FileNotFoundError: If the dataset does not exist.
            ValueError: If the dataset is invalid.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Traffic dataset not found: {path}",
            )

        dataframe = pd.read_csv(path)

        if dataframe.empty:
            raise ValueError(
                f"Traffic dataset is empty: {path}",
            )

        dataframe = self._normalize_timestamp_column(
            dataframe,
        )

        self._validate_dataset(dataframe)

        return dataframe

    @staticmethod
    def _normalize_timestamp_column(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Normalize the timestamp column.

        Supports datasets where the timestamp was stored as the
        first CSV column, such as METR-LA.

        Args:
            dataframe: Raw traffic dataframe.

        Returns:
            Dataframe with a normalized ``timestamp`` column.

        Raises:
            ValueError: If no valid timestamp column is found.
        """
        if "timestamp" in dataframe.columns:
            return dataframe

        first_column = dataframe.columns[0]

        # Numeric sensor values must never be interpreted as timestamps.
        if pd.api.types.is_numeric_dtype(
            dataframe[first_column],
        ):
            raise ValueError(
                "Dataset must contain a valid timestamp column.",
            )

        parsed_timestamps = pd.to_datetime(
            dataframe[first_column],
            errors="coerce",
        )

        valid_timestamp_ratio = float(
            parsed_timestamps.notna().mean(),
        )

        if valid_timestamp_ratio < 0.99:
            raise ValueError(
                "Dataset must contain a valid timestamp column.",
            )

        normalized = dataframe.copy()

        normalized = normalized.rename(
            columns={
                first_column: "timestamp",
            },
        )

        normalized["timestamp"] = parsed_timestamps

        return normalized

    @staticmethod
    def _validate_dataset(
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate the basic traffic dataset structure.

        Args:
            dataframe: Traffic dataframe.

        Raises:
            ValueError: If the dataset structure is invalid.
        """
        if "timestamp" not in dataframe.columns:
            raise ValueError(
                "Dataset must contain a timestamp column.",
            )

        if len(dataframe.columns) < 2:
            raise ValueError(
                "Dataset must contain at least one sensor column.",
            )

        timestamps = pd.to_datetime(
            dataframe["timestamp"],
            errors="coerce",
        )

        if timestamps.isna().any():
            raise ValueError(
                "Dataset timestamp column contains invalid timestamps.",
            )

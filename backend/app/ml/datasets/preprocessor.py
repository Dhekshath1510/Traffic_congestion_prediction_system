from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for traffic dataset preprocessing."""

    timestamp_column: str = "timestamp"

    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    forward_fill_limit: int = 3


@dataclass(frozen=True)
class DatasetSplit:
    """Chronological train, validation, and test datasets."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


class TrafficDatasetPreprocessor:
    """Preprocess and split traffic time-series data."""

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
    ) -> None:
        self._config = config or PreprocessingConfig()

        self._validate_configuration()

        self._fill_values: pd.Series | None = None

    def split(
        self,
        dataframe: pd.DataFrame,
    ) -> DatasetSplit:
        """Create chronological train/validation/test splits.

        Args:
            dataframe: Validated traffic dataframe.

        Returns:
            Chronological dataset splits.
        """
        timestamp_column = self._config.timestamp_column

        if timestamp_column not in dataframe.columns:
            raise ValueError(
                f"Dataset must contain '{timestamp_column}'.",
            )

        prepared = dataframe.copy()

        prepared[timestamp_column] = pd.to_datetime(
            prepared[timestamp_column],
            errors="coerce",
        )

        if prepared[timestamp_column].isna().any():
            raise ValueError(
                "Dataset contains invalid timestamps.",
            )

        prepared = (
            prepared.sort_values(timestamp_column)
            .drop_duplicates(
                subset=[timestamp_column],
                keep="first",
            )
            .reset_index(drop=True)
        )

        row_count = len(prepared)

        train_end = int(
            row_count * self._config.train_ratio,
        )

        validation_end = train_end + int(
            row_count * self._config.validation_ratio,
        )

        train = prepared.iloc[:train_end].copy()

        validation = prepared.iloc[train_end:validation_end].copy()

        test = prepared.iloc[validation_end:].copy()

        return DatasetSplit(
            train=train,
            validation=validation,
            test=test,
        )

    def fit(
        self,
        train_dataframe: pd.DataFrame,
    ) -> None:
        """Fit preprocessing parameters using training data only.

        Args:
            train_dataframe: Training dataset.
        """
        sensor_columns = self._get_sensor_columns(
            train_dataframe,
        )

        numeric_training_data = train_dataframe[sensor_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )

        self._fill_values = numeric_training_data.median()

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform a dataset using fitted training parameters.

        Args:
            dataframe: Dataset to transform.

        Returns:
            Preprocessed dataset.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._fill_values is None:
            raise RuntimeError(
                "Preprocessor must be fitted before transform().",
            )

        result = dataframe.copy()

        sensor_columns = self._get_sensor_columns(
            result,
        )

        result[sensor_columns] = (
            result[sensor_columns]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .ffill(
                limit=self._config.forward_fill_limit,
            )
            .fillna(self._fill_values)
        )

        return result

    def fit_transform(
        self,
        train_dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fit preprocessing parameters and transform training data.

        Args:
            train_dataframe: Training dataset.

        Returns:
            Transformed training dataset.
        """
        self.fit(train_dataframe)

        return self.transform(train_dataframe)

    def _get_sensor_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> list[str]:
        """Return all traffic sensor columns."""
        timestamp_column = self._config.timestamp_column

        return [column for column in dataframe.columns if column != timestamp_column]

    def _validate_configuration(self) -> None:
        """Validate preprocessing configuration."""
        ratios = (
            self._config.train_ratio
            + self._config.validation_ratio
            + self._config.test_ratio
        )

        if abs(ratios - 1.0) > 1e-9:
            raise ValueError(
                "Train, validation, and test ratios must sum to 1.",
            )

        if not 0 < self._config.train_ratio < 1:
            raise ValueError(
                "train_ratio must be between 0 and 1.",
            )

        if not 0 < self._config.validation_ratio < 1:
            raise ValueError(
                "validation_ratio must be between 0 and 1.",
            )

        if not 0 < self._config.test_ratio < 1:
            raise ValueError(
                "test_ratio must be between 0 and 1.",
            )

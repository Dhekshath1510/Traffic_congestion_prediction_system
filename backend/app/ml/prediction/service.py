from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

import joblib
import numpy as np
import pandas as pd
import torch

from app.ml.classification.congestion_classifier import (
    CongestionClassifier,
)
from app.ml.explainability.shap_explainer import (
    CLASS_NAMES,
    CongestionSHAPExplainer,
    SHAPExplanation,
)
from app.ml.models.gru_model import (
    GRUConfig,
    TrafficGRUModel,
)


# ============================================================================
# Constants
# ============================================================================

HISTORY_STEPS = 12
SENSOR_COUNT = 207
CLASS_COUNT = 3

# The classifier was trained using these 25 features.
CLASSIFIER_FEATURE_NAMES: tuple[str, ...] = (
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
    "sensor_index",
)

LOW_THRESHOLD = 0.10
SEVERE_THRESHOLD = 0.30


# ============================================================================
# Exceptions
# ============================================================================


class PredictionServiceError(RuntimeError):
    """Base exception for prediction-service failures."""


class ArtifactLoadError(PredictionServiceError):
    """Raised when a required ML artifact cannot be loaded."""


class InvalidPredictionInput(PredictionServiceError):
    """Raised when prediction input is invalid."""


# ============================================================================
# Dependency Protocols
# ============================================================================


class TrafficForecaster(Protocol):
    """Protocol for a traffic-speed forecasting model."""

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Predict future speed for all sensors."""
        ...


class CongestionModel(Protocol):
    """Protocol required from the congestion classifier."""

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Predict congestion class."""
        ...

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Predict congestion probabilities."""
        ...


class Explainer(Protocol):
    """Protocol for a single-instance explanation provider."""

    def explain_instance(
        self,
        X: np.ndarray,
        instance_index: int = 0,
        top_n: int = 10,
    ) -> SHAPExplanation:
        """Explain one classifier prediction."""
        ...


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class PredictionServiceConfig:
    """
    Configuration for production traffic prediction.

    Paths are relative to the project root unless absolute paths are supplied.
    """

    gru_checkpoint: Path = Path(
        "results/models/gru_residual_model.pt"
    )

    classifier_checkpoint: Path = Path(
        "results/models/congestion_classifier.joblib"
    )

    reference_speed_file: Path = Path(
        "results/classification/sensor_reference_speeds.csv"
    )

    device: str = "cpu"

    shap_top_n: int = 10


# ============================================================================
# Prediction Result
# ============================================================================


@dataclass(frozen=True)
class PredictionResult:
    """Complete result returned by the prediction service."""

    sensor_id: int
    timestamp: datetime

    predicted_speed: float
    reference_speed: float
    congestion_index: float

    congestion_level: str
    confidence: float

    probabilities: dict[str, float]

    explanation: SHAPExplanation


# ============================================================================
# Prediction Service
# ============================================================================


class PredictionService:
    """
    Application-level orchestration service for traffic prediction.

    Pipeline:

        12 x 207 historical speeds
                    |
                    v
              GRU forecasting
                    |
                    v
          Future speed for 207 sensors
                    |
                    v
          Requested sensor prediction
                    |
                    v
        Sensor-specific reference speed
                    |
                    v
            Congestion Index
                    |
                    v
       LOW / MODERATE / SEVERE
                    |
                    v
       XGBoost classification
                    |
                    v
                TreeSHAP

    The service intentionally contains no FastAPI-specific logic.
    """

    def __init__(
        self,
        config: PredictionServiceConfig | None = None,
        *,
        forecaster: TrafficForecaster | None = None,
        classifier: CongestionModel | None = None,
        explainer: Explainer | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else PredictionServiceConfig()
        )

        self._forecaster = forecaster
        self._classifier = classifier
        self._explainer = explainer

        self._reference_speeds: dict[int, float] | None = None

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def predict(
        self,
        *,
        sensor_id: int,
        timestamp: datetime,
        recent_speeds: list[list[float]],
    ) -> PredictionResult:
        """
        Generate one congestion prediction.

        Args:
            sensor_id:
                External sensor identifier.

            timestamp:
                Timestamp associated with the latest observation.

            recent_speeds:
                12 historical observations for all 207 sensors.

                Shape:
                    (12, 207)

        Returns:
            PredictionResult containing:

            - predicted future speed
            - reference speed
            - congestion index
            - congestion class
            - confidence
            - class probabilities
            - SHAP explanation
        """

        history = self._prepare_history(
            recent_speeds
        )

        sensor_index = self._resolve_sensor_index(
            sensor_id
        )

        future_speeds = self._get_forecaster().predict(
            history[np.newaxis, :, :]
        )

        self._validate_forecast(
            future_speeds
        )

        predicted_speed = float(
            future_speeds[
                0,
                sensor_index,
            ]
        )

        reference_speed = (
            self._get_reference_speed(
                sensor_index
            )
        )

        congestion_index = (
            self._calculate_congestion_index(
                predicted_speed=predicted_speed,
                reference_speed=reference_speed,
            )
        )

        classifier_features = (
            self._build_classifier_features(
                history=history,
                sensor_index=sensor_index,
                timestamp=timestamp,
            )
        )

        classifier_prediction = (
            self._get_classifier().predict(
                classifier_features
            )
        )

        probabilities = (
            self._get_classifier().predict_proba(
                classifier_features
            )
        )

        self._validate_classifier_output(
            classifier_prediction,
            probabilities,
        )

        predicted_class_id = int(
            classifier_prediction[0]
        )

        probability_vector = probabilities[0]

        confidence = float(
            np.max(probability_vector)
        )

        probability_dict = {
            CLASS_NAMES[index]: float(
                probability_vector[index]
            )
            for index in range(CLASS_COUNT)
        }

        explanation = (
            self._get_explainer().explain_instance(
                classifier_features,
                instance_index=0,
                top_n=self.config.shap_top_n,
            )
        )

        return PredictionResult(
            sensor_id=sensor_id,
            timestamp=timestamp,
            predicted_speed=predicted_speed,
            reference_speed=reference_speed,
            congestion_index=congestion_index,
            congestion_level=CLASS_NAMES[
                predicted_class_id
            ],
            confidence=confidence,
            probabilities=probability_dict,
            explanation=explanation,
        )

    # ----------------------------------------------------------------------
    # Model loading
    # ----------------------------------------------------------------------

    def _get_forecaster(
        self,
    ) -> TrafficForecaster:
        """Load the GRU model lazily and cache it."""

        if self._forecaster is not None:
            return self._forecaster

        checkpoint_path = (
            self.config.gru_checkpoint
        )

        if not checkpoint_path.exists():
            raise ArtifactLoadError(
                "GRU checkpoint was not found: "
                f"{checkpoint_path}"
            )

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.config.device,
                weights_only=False,
            )
        except Exception as exc:
            raise ArtifactLoadError(
                "Failed to load GRU checkpoint: "
                f"{checkpoint_path}"
            ) from exc

        if not isinstance(
            checkpoint,
            dict,
        ):
            raise ArtifactLoadError(
                "GRU checkpoint must contain a dictionary."
            )

        checkpoint = cast(
            dict[str, Any],
            checkpoint,
        )

        if "model_state_dict" not in checkpoint:
            raise ArtifactLoadError(
                "GRU checkpoint does not contain "
                "'model_state_dict'."
            )

        checkpoint_config = checkpoint.get(
            "config"
        )

        try:
            gru_config = (
                self._build_gru_config(
                    checkpoint_config
                )
            )

            model = TrafficGRUModel(
                config=gru_config,
                device=self.config.device,
            )

            model.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            setattr(
                model,
                "_trained",
                True,
            )

            model.model.eval()

        except Exception as exc:
            raise ArtifactLoadError(
                "Failed to initialize GRU from checkpoint."
            ) from exc

        self._forecaster = model

        return model

    def _get_classifier(
        self,
    ) -> CongestionModel:
        """Load and cache the trained congestion classifier."""

        if self._classifier is not None:
            return self._classifier

        checkpoint_path = (
            self.config.classifier_checkpoint
        )

        if not checkpoint_path.exists():
            raise ArtifactLoadError(
                "Congestion classifier checkpoint "
                f"was not found: {checkpoint_path}"
            )

        try:
            loaded_model = joblib.load(
                checkpoint_path
            )
        except Exception as exc:
            raise ArtifactLoadError(
                "Failed to load congestion classifier: "
                f"{checkpoint_path}"
            ) from exc

        if not isinstance(
            loaded_model,
            CongestionClassifier,
        ):
            raise ArtifactLoadError(
                "The classifier artifact is not a "
                "CongestionClassifier instance."
            )

        self._classifier = loaded_model

        return loaded_model

    def _get_explainer(
        self,
    ) -> Explainer:
        """Create and cache the TreeSHAP explainer."""

        if self._explainer is not None:
            return self._explainer

        classifier = self._get_classifier()

        try:
            explainer = CongestionSHAPExplainer(
                model=classifier,
                feature_names=list(
                    CLASSIFIER_FEATURE_NAMES
                ),
            )
        except Exception as exc:
            raise ArtifactLoadError(
                "Failed to initialize TreeSHAP explainer."
            ) from exc

        self._explainer = explainer

        return explainer

    # ----------------------------------------------------------------------
    # GRU configuration
    # ----------------------------------------------------------------------

    @staticmethod
    def _build_gru_config(
        raw_config: object,
    ) -> GRUConfig:
        """
        Convert checkpoint configuration into GRUConfig.

        The saved checkpoint contains the configuration dictionary
        generated during training.
        """

        if not isinstance(
            raw_config,
            dict,
        ):
            return GRUConfig()

        allowed_fields = {
            "input_size",
            "hidden_size",
            "num_layers",
            "dropout",
            "output_size",
        }

        values: dict[str, object] = {
            key: value
            for key, value in cast(
                dict[str, object],
                raw_config,
            ).items()
            if key in allowed_fields
        }

        try:
            return GRUConfig(
                input_size=int(
                    cast(
                        Any,
                        values.get(
                            "input_size",
                            207,
                        ),
                    )
                ),
                hidden_size=int(
                    cast(
                        Any,
                        values.get(
                            "hidden_size",
                            64,
                        ),
                    )
                ),
                num_layers=int(
                    cast(
                        Any,
                        values.get(
                            "num_layers",
                            1,
                        ),
                    )
                ),
                dropout=float(
                    cast(
                        Any,
                        values.get(
                            "dropout",
                            0.0,
                        ),
                    )
                ),
                output_size=int(
                    cast(
                        Any,
                        values.get(
                            "output_size",
                            207,
                        ),
                    )
                ),
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ArtifactLoadError(
                "Invalid GRU configuration stored "
                "inside checkpoint."
            ) from exc

    # ----------------------------------------------------------------------
    # Input preparation
    # ----------------------------------------------------------------------

    @staticmethod
    def _prepare_history(
        recent_speeds: list[list[float]],
    ) -> np.ndarray:
        """Validate and convert API history into a GRU input tensor."""

        history = np.asarray(
            recent_speeds,
            dtype=np.float32,
        )

        expected_shape = (
            HISTORY_STEPS,
            SENSOR_COUNT,
        )

        if history.shape != expected_shape:
            raise InvalidPredictionInput(
                "recent_speeds must have shape "
                f"{expected_shape}; received "
                f"{history.shape}."
            )

        if not np.isfinite(
            history
        ).all():
            raise InvalidPredictionInput(
                "recent_speeds contains NaN "
                "or infinite values."
            )

        if np.any(
            history < 0
        ):
            raise InvalidPredictionInput(
                "Traffic speeds cannot be negative."
            )

        return history

    # ----------------------------------------------------------------------
    # Sensor mapping
    # ----------------------------------------------------------------------

    @staticmethod
    def _resolve_sensor_index(
        sensor_id: int,
    ) -> int:
        """
        Resolve the external sensor identifier to the network index.

        The current API contract uses the sensor index as the identifier
        when no separate sensor-ID mapping artifact is available.

        The valid network indices are 0..206.

        If your API receives METR-LA sensor IDs such as 773869, this method
        must be replaced by a persistent sensor-ID-to-index repository.
        """

        if not 0 <= sensor_id < SENSOR_COUNT:
            raise InvalidPredictionInput(
                "sensor_id must currently represent the "
                f"zero-based sensor index 0..{SENSOR_COUNT - 1}."
            )

        return sensor_id

    # ----------------------------------------------------------------------
    # Reference speeds
    # ----------------------------------------------------------------------

    def _load_reference_speeds(
        self,
    ) -> dict[int, float]:
        """Load sensor-specific P85 reference speeds."""

        if self._reference_speeds is not None:
            return self._reference_speeds

        path = (
            self.config.reference_speed_file
        )

        if not path.exists():
            raise ArtifactLoadError(
                "Reference-speed file was not found: "
                f"{path}"
            )

        try:
            dataframe = pd.read_csv(
                path
            )
        except Exception as exc:
            raise ArtifactLoadError(
                "Failed to read reference-speed file: "
                f"{path}"
            ) from exc

        required_columns = {
            "sensor_index",
            "reference_speed",
        }

        if not required_columns.issubset(
            dataframe.columns
        ):
            raise ArtifactLoadError(
                "Reference-speed file must contain "
                "'sensor_index' and 'reference_speed'."
            )

        if dataframe.empty:
            raise ArtifactLoadError(
                "Reference-speed file is empty."
            )

        reference_speeds: dict[int, float] = {}

        for row in dataframe.itertuples(
            index=False
        ):
            sensor_index_one_based = int(
                getattr(
                    row,
                    "sensor_index",
                )
            )

            reference_speed = float(
                getattr(
                    row,
                    "reference_speed",
                )
            )

            if not np.isfinite(
                reference_speed
            ):
                raise ArtifactLoadError(
                    "Reference-speed artifact contains "
                    "a non-finite value."
                )

            if reference_speed <= 0:
                raise ArtifactLoadError(
                    "Reference speeds must be greater than zero."
                )

            reference_speeds[
                sensor_index_one_based
            ] = reference_speed

        if len(reference_speeds) != SENSOR_COUNT:
            raise ArtifactLoadError(
                "Reference-speed artifact contains "
                f"{len(reference_speeds)} sensors; "
                f"expected {SENSOR_COUNT}."
            )

        self._reference_speeds = (
            reference_speeds
        )

        return reference_speeds

    def _get_reference_speed(
        self,
        sensor_index: int,
    ) -> float:
        """
        Retrieve the P85 reference speed.

        The CSV uses 1-based sensor indices, while the classifier
        feature uses a zero-based sensor index.
        """

        reference_speeds = (
            self._load_reference_speeds()
        )

        csv_sensor_index = (
            sensor_index + 1
        )

        try:
            return reference_speeds[
                csv_sensor_index
            ]
        except KeyError as exc:
            raise ArtifactLoadError(
                "No reference speed exists for "
                f"sensor index {sensor_index}."
            ) from exc

    # ----------------------------------------------------------------------
    # Congestion index
    # ----------------------------------------------------------------------

    @staticmethod
    def _calculate_congestion_index(
        *,
        predicted_speed: float,
        reference_speed: float,
    ) -> float:
        """
        Calculate the sensor-relative congestion index.

            CI = clip(1 - future_speed / reference_speed, 0, 1)
        """

        if not np.isfinite(
            predicted_speed
        ):
            raise PredictionServiceError(
                "Predicted speed is not finite."
            )

        if not np.isfinite(
            reference_speed
        ) or reference_speed <= 0:
            raise PredictionServiceError(
                "Reference speed must be finite "
                "and greater than zero."
            )

        congestion_index = (
            1.0
            - (
                predicted_speed
                / reference_speed
            )
        )

        return float(
            np.clip(
                congestion_index,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _classify_congestion_index(
        congestion_index: float,
    ) -> str:
        """Convert CI into LOW/MODERATE/SEVERE."""

        if not np.isfinite(
            congestion_index
        ):
            raise PredictionServiceError(
                "Congestion index is not finite."
            )

        if congestion_index < LOW_THRESHOLD:
            return "LOW"

        if congestion_index < SEVERE_THRESHOLD:
            return "MODERATE"

        return "SEVERE"

    # ----------------------------------------------------------------------
    # Classifier feature construction
    # ----------------------------------------------------------------------

    @staticmethod
    def _build_classifier_features(
        *,
        history: np.ndarray,
        sensor_index: int,
        timestamp: datetime,
    ) -> np.ndarray:
        """
        Build the 25-dimensional classifier feature vector.

        Features reproduce the classifier feature schema used in
        the congestion-classification experiment.

        Output shape:

            (1, 25)
        """

        if history.shape != (
            HISTORY_STEPS,
            SENSOR_COUNT,
        ):
            raise InvalidPredictionInput(
                "Unexpected history shape while "
                "building classifier features."
            )

        sensor_history = history[
            :,
            sensor_index,
        ].astype(
            np.float32,
            copy=False,
        )

        current_speed = float(
            sensor_history[-1]
        )

        previous_speed = float(
            sensor_history[-2]
        )

        speed_changes = np.diff(
            sensor_history
        )

        long_term_speed_change = (
            current_speed
            - float(sensor_history[0])
        )

        sequence_mean = float(
            np.mean(
                sensor_history
            )
        )

        sequence_std = float(
            np.std(
                sensor_history
            )
        )

        # The classifier uses recent traffic statistics.
        # The most recent four observations correspond to the
        # final 20 minutes of the 5-minute traffic sequence.
        recent_window = sensor_history[
            -4:
        ]

        recent_mean = float(
            np.mean(
                recent_window
            )
        )

        recent_std = float(
            np.std(
                recent_window
            )
        )

        near_zero_fraction = float(
            np.mean(
                sensor_history <= 1.0
            )
        )

        low_speed_fraction = float(
            np.mean(
                sensor_history < 30.0
            )
        )

        moderate_speed_fraction = float(
            np.mean(
                (
                    sensor_history >= 30.0
                )
                & (
                    sensor_history < 50.0
                )
            )
        )

        high_speed_fraction = float(
            np.mean(
                sensor_history >= 50.0
            )
        )

        timestamp_value = pd.Timestamp(
            timestamp
        )

        hour = float(
            timestamp_value.hour
            + (
                timestamp_value.minute
                / 60.0
            )
        )

        hour_angle = (
            2.0
            * np.pi
            * hour
            / 24.0
        )

        hour_sin = float(
            np.sin(
                hour_angle
            )
        )

        hour_cos = float(
            np.cos(
                hour_angle
            )
        )

        day_of_week = float(
            timestamp_value.dayofweek
        )

        day_angle = (
            2.0
            * np.pi
            * day_of_week
            / 7.0
        )

        day_sin = float(
            np.sin(
                day_angle
            )
        )

        day_cos = float(
            np.cos(
                day_angle
            )
        )

        is_weekend = float(
            timestamp_value.dayofweek >= 5
        )

        feature_vector = np.array(
            [
                current_speed,
                float(
                    np.std(
                        np.array(
                            [current_speed],
                            dtype=np.float32,
                        )
                    )
                ),
                current_speed,
                current_speed,
                previous_speed,
                float(
                    np.std(
                        np.array(
                            [previous_speed],
                            dtype=np.float32,
                        )
                    )
                ),
                float(
                    np.mean(
                        speed_changes
                    )
                ),
                float(
                    np.std(
                        speed_changes
                    )
                ),
                long_term_speed_change,
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
                float(sensor_index),
            ],
            dtype=np.float32,
        )

        if feature_vector.shape != (
            len(CLASSIFIER_FEATURE_NAMES),
        ):
            raise PredictionServiceError(
                "Classifier feature construction produced "
                f"{feature_vector.shape[0]} features; "
                f"expected {len(CLASSIFIER_FEATURE_NAMES)}."
            )

        if not np.isfinite(
            feature_vector
        ).all():
            raise PredictionServiceError(
                "Classifier feature vector contains "
                "NaN or infinite values."
            )

        return feature_vector.reshape(
            1,
            -1,
        )

    # ----------------------------------------------------------------------
    # Output validation
    # ----------------------------------------------------------------------

    @staticmethod
    def _validate_forecast(
        forecast: np.ndarray,
    ) -> None:
        """Validate GRU output."""

        if forecast.ndim != 2:
            raise PredictionServiceError(
                "GRU forecast must be 2-dimensional."
            )

        if forecast.shape != (
            1,
            SENSOR_COUNT,
        ):
            raise PredictionServiceError(
                "Unexpected GRU output shape: "
                f"{forecast.shape}; expected "
                f"(1, {SENSOR_COUNT})."
            )

        if not np.isfinite(
            forecast
        ).all():
            raise PredictionServiceError(
                "GRU forecast contains NaN "
                "or infinite values."
            )

    @staticmethod
    def _validate_classifier_output(
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> None:
        """Validate classifier prediction and probability output."""

        if predictions.shape != (
            1,
        ):
            raise PredictionServiceError(
                "Unexpected classifier prediction shape: "
                f"{predictions.shape}."
            )

        if probabilities.shape != (
            1,
            CLASS_COUNT,
        ):
            raise PredictionServiceError(
                "Unexpected classifier probability shape: "
                f"{probabilities.shape}; expected "
                f"(1, {CLASS_COUNT})."
            )

        if not np.isfinite(
            probabilities
        ).all():
            raise PredictionServiceError(
                "Classifier probabilities contain "
                "NaN or infinite values."
            )

        if np.any(
            probabilities < 0
        ):
            raise PredictionServiceError(
                "Classifier probabilities cannot be negative."
            )

        probability_sum = float(
            np.sum(
                probabilities[0]
            )
        )

        if not np.isclose(
            probability_sum,
            1.0,
            atol=1e-4,
        ):
            raise PredictionServiceError(
                "Classifier probabilities do not "
                f"sum to one: {probability_sum:.6f}."
            )

        predicted_class = int(
            predictions[0]
        )

        if not 0 <= predicted_class < CLASS_COUNT:
            raise PredictionServiceError(
                "Classifier returned an invalid "
                f"class ID: {predicted_class}."
            )
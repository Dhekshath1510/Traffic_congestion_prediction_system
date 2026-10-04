from __future__ import annotations
from pathlib import Path
import numpy as np
from app.ml.classification.feature_builder import (
    EXPECTED_SENSOR_COUNT,
    build_classifier_features,
    CLASSIFIER_FEATURE_NAMES,
)
from app.ml.explainability.shap_explainer import (
    CongestionSHAPExplainer,
)
from app.ml.runtime.model_registry import ModelRegistry
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    SensorPrediction,
    SHAPExplanationResponse,
    SHAPFeatureContribution,
)


CLASS_NAMES = (
    "LOW",
    "MODERATE",
    "SEVERE",
)


class ReferenceSpeedProvider:
    """Provides sensor-specific training reference speeds."""

    def __init__(self, path: Path) -> None:
        self._speeds = self._load(path)

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(
                f"Reference speed artifact not found: {path}"
            )

        data = np.genfromtxt(
            path,
            delimiter=",",
            names=True,
        )

        if data.shape[0] != EXPECTED_SENSOR_COUNT:
            raise ValueError(
                "Unexpected number of reference speeds: "
                f"{data.shape[0]}. "
                f"Expected {EXPECTED_SENSOR_COUNT}."
            )

        speeds = np.asarray(
            data["reference_speed"],
            dtype=np.float32,
        )

        if not np.isfinite(speeds).all():
            raise ValueError(
                "Reference speeds contain non-finite values."
            )

        if np.any(speeds <= 0):
            raise ValueError(
                "Reference speeds must be positive."
            )

        return speeds

    def get_all(self) -> np.ndarray:
        return self._speeds.copy()


class PredictionService:
    """Coordinates traffic forecasting and congestion classification."""

    def __init__(
        self,
        registry: ModelRegistry,
        reference_speed_provider: ReferenceSpeedProvider,
    ) -> None:
        self._registry = registry
        self._reference_speeds = reference_speed_provider

        self._shap_explainer = CongestionSHAPExplainer(
            model=self._registry.classifier,
            feature_names=list(CLASSIFIER_FEATURE_NAMES),
        )

    def predict(
        self,
        request: PredictionRequest,
    ) -> PredictionResponse:
        sequence = np.asarray(
            request.sequence,
            dtype=np.float32,
        )

        expected_shape = (
            12,
            EXPECTED_SENSOR_COUNT,
        )

        if sequence.shape != expected_shape:
            raise ValueError(
                "Expected sequence shape "
                f"{expected_shape}, got {sequence.shape}."
            )

        if not np.isfinite(sequence).all():
            raise ValueError(
                "Sequence contains NaN or infinite values."
            )

        # ------------------------------------------------------------------
        # 1. Forecast future speed for all 207 sensors.
        # ------------------------------------------------------------------
        model_input = sequence[np.newaxis, :, :]

        predicted_speed = self._registry.gru.predict(
            model_input,
        )[0]

        # ------------------------------------------------------------------
        # 2. Calculate sensor-relative congestion index.
        #
        # CI = clip(1 - future_speed / reference_speed, 0, 1)
        # ------------------------------------------------------------------
        reference_speeds = (
            self._reference_speeds.get_all()
        )

        congestion_index = np.clip(
            1.0
            - (
                predicted_speed
                / reference_speeds
            ),
            0.0,
            1.0,
        )

        # ------------------------------------------------------------------
        # 3. Build the exact 25 classifier features used during training.
        # ------------------------------------------------------------------
        classifier_features = build_classifier_features(
            sequence=sequence,
            timestamp=request.timestamp,
            gru_predicted_speed=predicted_speed,
        )

        # ------------------------------------------------------------------
        # 4. Classify congestion for each sensor.
        # ------------------------------------------------------------------
        class_ids = self._registry.classifier.predict(
            classifier_features,
        )

        probabilities = (
            self._registry.classifier.predict_proba(
                classifier_features,
            )
        )

        # ------------------------------------------------------------------
        # 5. Generate SHAP explanations for the same XGBoost inputs.
        # ------------------------------------------------------------------

        shap_values = self._shap_explainer.calculate(
            classifier_features,
        )
        # ------------------------------------------------------------------
        # 5. Build API response.
        # ------------------------------------------------------------------
        sensors: list[SensorPrediction] = []

        for sensor_index in range(
            EXPECTED_SENSOR_COUNT
        ):
            class_id = int(
                class_ids[sensor_index]
            )

            sensor_shap_values = shap_values[
                sensor_index,
                :,
                class_id,
            ]

            ranking = np.argsort(
                np.abs(sensor_shap_values)
            )[::-1]

            top_features = [
                SHAPFeatureContribution(
                    feature=CLASSIFIER_FEATURE_NAMES[index], # type: ignore
                    contribution=float(
                        sensor_shap_values[index]
                    ),
                )
                for index in ranking[:5]
            ]

            explanation = SHAPExplanationResponse(
                predicted_class=CLASS_NAMES[class_id],
                predicted_class_id=class_id,
                top_features=top_features,
            )

            sensors.append(
                SensorPrediction(
                    sensor_index=sensor_index,
                    predicted_speed=float(
                        predicted_speed[
                            sensor_index
                        ]
                    ),
                    congestion_index=float(
                        congestion_index[
                            sensor_index
                        ]
                    ),
                    congestion_class=CLASS_NAMES[
                        class_id
                    ],
                    probabilities={
                        CLASS_NAMES[0]: float(
                            probabilities[
                                sensor_index,
                                0,
                            ]
                        ),
                        CLASS_NAMES[1]: float(
                            probabilities[
                                sensor_index,
                                1,
                            ]
                        ),
                        CLASS_NAMES[2]: float(
                            probabilities[
                                sensor_index,
                                2,
                            ]
                        ),
                    },
                    explanation=explanation,
                )
            )

        return PredictionResponse(
            timestamp=request.timestamp,
            forecast_horizon="next 5 minutes",
            sensors=sensors,
        )
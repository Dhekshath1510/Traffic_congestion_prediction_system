from __future__ import annotations

import joblib

from app.ml.classification.congestion_classifier import (
    CongestionClassifier,
)
from app.ml.models.gru_model import TrafficGRUModel
from app.ml.runtime.artifacts import ModelArtifactPaths


class ModelRegistry:
    """
    Application-scoped registry for trained ML models.

    Models are loaded once and reused across prediction requests.
    """

    def __init__(
        self,
        artifact_paths: ModelArtifactPaths,
    ) -> None:
        self._artifact_paths = artifact_paths

        self._gru: TrafficGRUModel | None = None
        self._classifier: CongestionClassifier | None = None

    def load(self) -> None:
        """Load and validate all production models."""

        self._artifact_paths.validate()

        self._load_gru()
        self._load_classifier()

    def _load_gru(self) -> None:
        """Load the trained GRU forecasting model."""

        model = TrafficGRUModel()

        model.load_checkpoint(
            self._artifact_paths.gru_checkpoint,
        )

        self._gru = model

    def _load_classifier(self) -> None:
        """Load and validate the trained congestion classifier."""

        loaded = joblib.load(
            self._artifact_paths.congestion_classifier,
        )

        if not isinstance(
            loaded,
            CongestionClassifier,
        ):
            raise TypeError(
                "Invalid congestion classifier artifact. "
                f"Expected CongestionClassifier, got "
                f"{type(loaded)!r}."
            )

        if loaded.model.n_features_in_ != 26:
            raise ValueError(
                "Congestion classifier expects 26 features, "
                f"got {loaded.model.n_features_in_}."
            )

        if len(loaded.model.classes_) != 3:
            raise ValueError(
                "Invalid congestion classifier class count: "
                f"{len(loaded.model.classes_)}. Expected 3."
            )

        self._classifier = loaded

    @property
    def gru(self) -> TrafficGRUModel:
        """Return the loaded GRU forecasting model."""

        if self._gru is None:
            raise RuntimeError(
                "GRU model has not been loaded."
            )

        return self._gru

    @property
    def classifier(self) -> CongestionClassifier:
        """Return the loaded congestion classifier."""

        if self._classifier is None:
            raise RuntimeError(
                "Congestion classifier has not been loaded."
            )

        return self._classifier
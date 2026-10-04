from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from xgboost import XGBClassifier


CLASS_NAMES = (
    "LOW",
    "MODERATE",
    "SEVERE",
)


@dataclass(frozen=True)
class CongestionClassifierConfig:
    """Configuration for the shared congestion classifier."""

    n_estimators: int = 150
    max_depth: int = 5
    learning_rate: float = 0.08
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    random_state: int = 42
    n_jobs: int = -1


class CongestionClassifier:
    """
    Shared multiclass congestion classifier.

    The model predicts one congestion state for one sensor observation.

    Input features:
        engineered traffic/temporal features + sensor identity

    Output:
        0 -> LOW
        1 -> MODERATE
        2 -> SEVERE
    """

    def __init__(
        self,
        config: CongestionClassifierConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else CongestionClassifierConfig()
        )

        self.model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            eval_metric="mlogloss",
            tree_method="hist",
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Train the classifier."""

        if X.ndim != 2:
            raise ValueError(
                f"Expected 2-D features, got {X.shape}"
            )

        if y.ndim != 1:
            raise ValueError(
                f"Expected 1-D labels, got {y.shape}"
            )

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "Feature/label sample mismatch: "
                f"{X.shape[0]} != {y.shape[0]}"
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "Features contain NaN or infinite values."
            )

        if sample_weight is not None:
            if sample_weight.ndim != 1:
                raise ValueError(
                    "Sample weights must be 1-D."
                )

            if len(sample_weight) != len(y):
                raise ValueError(
                    "Sample-weight length does not match labels."
                )

        self.model.fit(
            X,
            y,
            sample_weight=sample_weight,
        )

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Predict congestion class IDs."""

        self._validate_features(X)

        return self.model.predict(X).astype(
            np.int8,
        )

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Return LOW/MODERATE/SEVERE probabilities."""

        self._validate_features(X)

        probabilities = self.model.predict_proba(X)

        if probabilities.shape[1] != 3:
            raise RuntimeError(
                "Classifier did not return three class probabilities."
            )

        return probabilities.astype(
            np.float32,
        )

    def _validate_features(
        self,
        X: np.ndarray,
    ) -> None:
        if X.ndim != 2:
            raise ValueError(
                f"Expected 2-D features, got {X.shape}"
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "Features contain NaN or infinite values."
            )

    def get_feature_importance(self) -> np.ndarray:
        """Return XGBoost gain-based feature importance."""

        return self.model.feature_importances_

    def get_booster_config(self) -> dict[str, Any]:
        """Return model configuration for reproducibility."""

        return {
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "min_child_weight": self.config.min_child_weight,
            "random_state": self.config.random_state,
            "n_jobs": self.config.n_jobs,
            "objective": "multi:softprob",
            "num_class": 3,
            "tree_method": "hist",
        }
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBRegressor

from app.ml.models.config import XGBoostConfig


class TrafficXGBoostModel:
    """Lightweight XGBoost traffic forecasting model."""

    def __init__(
        self,
        config: XGBoostConfig | None = None,
        *,
        n_estimators: int | None = None,
        max_depth: int | None = None,
        learning_rate: float | None = None,
        subsample: float | None = None,
        colsample_bytree: float | None = None,
        random_state: int | None = None,
    ) -> None:
        """Initialize the XGBoost model."""
        model_config = config or XGBoostConfig()

        if n_estimators is not None:
            model_config = replace(
                model_config,
                n_estimators=n_estimators,
            )

        if max_depth is not None:
            model_config = replace(
                model_config,
                max_depth=max_depth,
            )

        if learning_rate is not None:
            model_config = replace(
                model_config,
                learning_rate=learning_rate,
            )

        if subsample is not None:
            model_config = replace(
                model_config,
                subsample=subsample,
            )

        if colsample_bytree is not None:
            model_config = replace(
                model_config,
                colsample_bytree=colsample_bytree,
            )

        if random_state is not None:
            model_config = replace(
                model_config,
                random_state=random_state,
            )

        self._model = XGBRegressor(
            n_estimators=model_config.n_estimators,
            max_depth=model_config.max_depth,
            learning_rate=model_config.learning_rate,
            subsample=model_config.subsample,
            colsample_bytree=model_config.colsample_bytree,
            objective="reg:squarederror",
            tree_method="hist",
            max_bin=64,
            n_jobs=-1,
            random_state=model_config.random_state,
        )

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """Train a single-target XGBoost model."""
        X = np.asarray(
            features,
            dtype=np.float32,
        )

        y = np.asarray(
            targets,
            dtype=np.float32,
        )

        if X.ndim != 2:
            raise ValueError(
                "Expected features with shape " "(samples, features).",
            )

        if y.ndim != 1:
            raise ValueError(
                "Expected targets with shape " "(samples,).",
            )

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "Feature and target sample counts " "must match.",
            )

        self._model.fit(
            X,
            y,
        )

    def predict(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """Generate single-target predictions."""
        X = np.asarray(
            features,
            dtype=np.float32,
        )

        if X.ndim != 2:
            raise ValueError(
                "Expected features with shape " "(samples, features).",
            )

        return np.asarray(
            self._model.predict(X),
            dtype=np.float32,
        )

    def save(
        self,
        path: Path,
    ) -> None:
        """Save the trained model."""
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self._model,
            path,
        )

    def load(
        self,
        path: Path,
    ) -> None:
        """Load the trained model."""
        self._model = joblib.load(path)

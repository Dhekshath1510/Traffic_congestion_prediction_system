import numpy as np


class PersistenceBaseline:
    """Naive traffic forecasting baseline."""

    def predict(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """Predict future traffic using the latest observation.

        Args:
            features:
                Array with shape
                (samples, history_steps, sensors).

        Returns:
            Predictions with shape
            (samples, sensors).
        """
        if features.ndim != 3:
            raise ValueError(
                "Expected features with shape " "(samples, history_steps, sensors).",
            )

        return features[:, -1, :]

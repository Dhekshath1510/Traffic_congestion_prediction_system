from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap


CLASS_NAMES = (
    "LOW",
    "MODERATE",
    "SEVERE",
)


@dataclass(frozen=True)
class SHAPExplanation:
    """Structured explanation for one congestion prediction."""

    predicted_class: str
    predicted_class_id: int
    probabilities: dict[str, float]
    feature_contributions: list[dict[str, Any]]


class CongestionSHAPExplainer:
    """
    SHAP explainer for the trained congestion XGBoost classifier.

    The explainer operates directly on the trained XGBoost model.
    No surrogate classifier is introduced.
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
    ) -> None:
        if not feature_names:
            raise ValueError(
                "feature_names cannot be empty."
            )

        self.model_wrapper = model

        if not hasattr(model, "model"):
            raise TypeError(
                "Expected a CongestionClassifier instance "
                "with a trained 'model' attribute."
            )

        self.model = model.model

        self.feature_names = list(
            feature_names
        )

        if not hasattr(
            self.model,
            "get_booster",
        ):
            raise TypeError(
                "The supplied classifier does not appear "
                "to be an XGBoost model."
            )

        self.explainer = shap.TreeExplainer(
            self.model
        )

    # ------------------------------------------------------------------
    # SHAP calculation
    # ------------------------------------------------------------------

    def calculate(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate SHAP values.

        Returns an array normalized to:

            (samples, features, classes)

        regardless of the SHAP version's native output format.
        """

        self._validate_input(X)

        explanation = self.explainer(
            X
        )

        values = explanation.values

        values = np.asarray(
            values
        )

        # Modern SHAP for multiclass tree models normally returns:
        #
        #     (samples, features, classes)
        #
        # Older versions may return a list:
        #
        #     [class_0_array, class_1_array, class_2_array]
        #
        # Normalize both forms.

        if isinstance(
            values,
            list,
        ):
            values = np.stack(
                values,
                axis=-1,
            )

        if values.ndim == 2:
            # Binary/single-output fallback.
            values = values[
                :,
                :,
                np.newaxis,
            ]

        if values.ndim != 3:
            raise RuntimeError(
                "Unexpected SHAP output shape: "
                f"{values.shape}"
            )

        if values.shape[0] != X.shape[0]:
            raise RuntimeError(
                "SHAP sample dimension does not match input."
            )

        if values.shape[1] != X.shape[1]:
            raise RuntimeError(
                "SHAP feature dimension does not match input."
            )

        return values.astype(
            np.float32,
            copy=False,
        )

    # ------------------------------------------------------------------
    # Global importance
    # ------------------------------------------------------------------

    def global_importance(
        self,
        X: np.ndarray,
    ) -> pd.DataFrame:
        """
        Calculate global mean absolute SHAP importance.

        For multiclass classification, importance is aggregated across
        all classes.
        """

        shap_values = self.calculate(
            X
        )

        importance = np.mean(
            np.abs(shap_values),
            axis=(0, 2),
        )

        dataframe = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": importance,
            }
        )

        return dataframe.sort_values(
            "mean_abs_shap",
            ascending=False,
        ).reset_index(
            drop=True
        )

    # ------------------------------------------------------------------
    # Class-specific importance
    # ------------------------------------------------------------------

    def class_importance(
        self,
        X: np.ndarray,
        class_id: int,
    ) -> pd.DataFrame:
        """
        Calculate mean absolute SHAP importance for one class.
        """

        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(
                f"Invalid class ID: {class_id}"
            )

        shap_values = self.calculate(
            X
        )

        importance = np.mean(
            np.abs(
                shap_values[:, :, class_id]
            ),
            axis=0,
        )

        dataframe = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": importance,
            }
        )

        dataframe["class"] = CLASS_NAMES[
            class_id
        ]

        return dataframe.sort_values(
            "mean_abs_shap",
            ascending=False,
        ).reset_index(
            drop=True
        )

    # ------------------------------------------------------------------
    # Single prediction explanation
    # ------------------------------------------------------------------

    def explain_instance(
        self,
        X: np.ndarray,
        instance_index: int = 0,
        top_n: int = 10,
    ) -> SHAPExplanation:
        """
        Generate a structured explanation for one prediction.
        """

        self._validate_input(X)

        if not 0 <= instance_index < len(X):
            raise IndexError(
                f"Instance index {instance_index} "
                f"is outside range 0..{len(X) - 1}."
            )

        if top_n <= 0:
            raise ValueError(
                "top_n must be greater than zero."
            )

        instance = X[
            instance_index:
            instance_index + 1
        ]

        prediction = self.model_wrapper.predict(
            instance
        )[0]

        probabilities = self.model_wrapper.predict_proba(
            instance
        )[0]

        shap_values = self.calculate(
            instance
        )[0]

        class_values = shap_values[
            :,
            prediction,
        ]

        dataframe = pd.DataFrame(
            {
                "feature": self.feature_names,
                "value": instance[0],
                "shap_value": class_values,
                "abs_shap": np.abs(
                    class_values
                ),
            }
        )

        dataframe = dataframe.sort_values(
            "abs_shap",
            ascending=False,
        ).head(
            top_n
        )

        contributions = []

        for row in dataframe.itertuples(
            index=False
        ):
            contributions.append(
                {
                    "feature": row.feature,
                    "value": float(row.value),
                    "shap_value": float(
                        row.shap_value
                    ),
                    "direction": (
                        "increases"
                        if row.shap_value > 0
                        else "decreases"
                    ),
                }
            )

        probability_dict = {
            CLASS_NAMES[index]: float(
                probabilities[index]
            )
            for index in range(
                len(CLASS_NAMES)
            )
        }

        return SHAPExplanation(
            predicted_class=CLASS_NAMES[
                prediction
            ],
            predicted_class_id=int(
                prediction
            ),
            probabilities=probability_dict,
            feature_contributions=contributions,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_input(
        self,
        X: np.ndarray,
    ) -> None:
        if X.ndim != 2:
            raise ValueError(
                f"Expected 2-D feature matrix, got {X.shape}"
            )

        if X.shape[1] != len(
            self.feature_names
        ):
            raise ValueError(
                "Feature count does not match feature names: "
                f"{X.shape[1]} != "
                f"{len(self.feature_names)}"
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "Input contains NaN or infinite values."
            )
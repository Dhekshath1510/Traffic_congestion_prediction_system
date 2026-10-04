from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from app.ml.datasets.sequence_loader import load_sequence_split
from app.ml.features.temporal import build_temporal_features
from app.ml.models.persistence import PersistenceBaseline
from app.ml.models.xgboost_model import TrafficXGBoostModel


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

MAX_TRAINING_SAMPLES = 8_000

# Train XGBoost for only a representative subset during development.
#
# The final experiment can increase this value to all sensors.
DEVELOPMENT_SENSOR_COUNT = 207

XGBOOST_ESTIMATORS = 30
XGBOOST_MAX_DEPTH = 4
XGBOOST_LEARNING_RATE = 0.1
XGBOOST_SUBSAMPLE = 0.7
XGBOOST_COLSAMPLE = 0.5


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def calculate_mae(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate mean absolute error."""
    return float(
        np.mean(
            np.abs(
                actual - predicted,
            ),
        ),
    )


def calculate_rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate root mean squared error."""
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    actual - predicted,
                ),
            ),
        ),
    )


def calculate_smape(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate symmetric mean absolute percentage error."""
    denominator = (
        np.abs(actual)
        + np.abs(predicted)
    )

    valid = denominator > 1e-8

    if not np.any(valid):
        return 0.0

    return float(
        np.mean(
            2.0
            * np.abs(
                actual[valid]
                - predicted[valid],
            )
            / denominator[valid],
        )
        * 100.0,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model_name: str,
    split_name: str,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> None:
    """Print evaluation metrics."""
    mae = calculate_mae(
        actual,
        predicted,
    )

    rmse = calculate_rmse(
        actual,
        predicted,
    )

    smape = calculate_smape(
        actual,
        predicted,
    )

    print()
    print(f"{model_name} — {split_name}")
    print("=" * 60)
    print(f"MAE:   {mae:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"sMAPE: {smape:.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the revised Sprint 3.3 experiment."""
    sequence_directory = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "sequences"
    )

    artifact_directory = (
        PROJECT_ROOT
        / "data"
        / "artifacts"
    )

    artifact_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading sequence datasets...")

    X_train, y_train = load_sequence_split(
        sequence_directory,
        "train",
    )

    X_validation, y_validation = (
        load_sequence_split(
            sequence_directory,
            "validation",
        )
    )

    X_test, y_test = load_sequence_split(
        sequence_directory,
        "test",
    )

    # -----------------------------------------------------------------------
    # Development sample limit
    # -----------------------------------------------------------------------

    if X_train.shape[0] > MAX_TRAINING_SAMPLES:
        X_train = X_train[
            :MAX_TRAINING_SAMPLES
        ]

        y_train = y_train[
            :MAX_TRAINING_SAMPLES
        ]

    print()
    print("Dataset")
    print("=======")
    print(f"Train:      {X_train.shape}")
    print(f"Train target: {y_train.shape}")
    print(f"Validation: {X_validation.shape}")
    print(f"Test:       {X_test.shape}")

    sensor_count = X_train.shape[2]

    sensor_count_for_xgboost = min(
        DEVELOPMENT_SENSOR_COUNT,
        sensor_count,
    )

    print()
    print(
        "XGBoost development sensors: "
        f"{sensor_count_for_xgboost}",
    )

    # -----------------------------------------------------------------------
    # Persistence baseline
    # -----------------------------------------------------------------------

    persistence = PersistenceBaseline()

    persistence_validation = (
        persistence.predict(
            X_validation,
        )
    )

    persistence_test = persistence.predict(
        X_test,
    )

    evaluate(
        "Persistence",
        "Validation",
        y_validation,
        persistence_validation,
    )

    evaluate(
        "Persistence",
        "Test",
        y_test,
        persistence_test,
    )

    # -----------------------------------------------------------------------
    # Compact temporal features
    # -----------------------------------------------------------------------

    print()
    print("Building compact temporal features...")

    X_train_tabular = build_temporal_features(
        X_train,
    )

    X_validation_tabular = build_temporal_features(
        X_validation,
    )

    X_test_tabular = build_temporal_features(
        X_test,
    )

    print(
        f"Tabular features: "
        f"{X_train_tabular.shape}",
    )

    # -----------------------------------------------------------------------
    # XGBoost
    # -----------------------------------------------------------------------

    print()
    print("Training lightweight XGBoost...")

    xgboost_predictions = np.empty(
        (
            X_test.shape[0],
            sensor_count_for_xgboost,
        ),
        dtype=np.float32,
    )

    xgboost_validation_predictions = np.empty(
        (
            X_validation.shape[0],
            sensor_count_for_xgboost,
        ),
        dtype=np.float32,
    )

    for sensor_index in range(
        sensor_count_for_xgboost,
    ):
        print(
            f"Training sensor "
            f"{sensor_index + 1}/"
            f"{sensor_count_for_xgboost}...",
        )

        feature_start = (
            sensor_index
        )

        feature_indices = np.array(
            [
                feature_start,
                sensor_count + feature_start,
                (2 * sensor_count)
                + feature_start,
                (3 * sensor_count)
                + feature_start,
            ],
            dtype=np.int64,
        )

        X_sensor_train = (
            X_train_tabular[
                :,
                feature_indices,
            ]
        )

        X_sensor_validation = (
            X_validation_tabular[
                :,
                feature_indices,
            ]
        )

        X_sensor_test = (
            X_test_tabular[
                :,
                feature_indices,
            ]
        )

        model = TrafficXGBoostModel(
            n_estimators=XGBOOST_ESTIMATORS,
            max_depth=XGBOOST_MAX_DEPTH,
            learning_rate=XGBOOST_LEARNING_RATE,
            subsample=XGBOOST_SUBSAMPLE,
            colsample_bytree=1.0,
            random_state=42,
        )

        model.fit(
            X_sensor_train,
            y_train[
                :,
                sensor_index,
            ],
        )

        xgboost_validation_predictions[
            :,
            sensor_index,
        ] = model.predict(
            X_sensor_validation,
        )

        xgboost_predictions[
            :,
            sensor_index,
        ] = model.predict(
            X_sensor_test,
        )

        model_path = (
            artifact_directory
            / (
                "xgboost_sensor_"
                f"{sensor_index}.joblib"
            )
        )

        model.save(
            model_path,
        )

    # -----------------------------------------------------------------------
    # Evaluate only the sensors trained by XGBoost
    # -----------------------------------------------------------------------

    evaluate(
        "XGBoost",
        "Validation",
        y_validation[
            :,
            :sensor_count_for_xgboost,
        ],
        xgboost_validation_predictions,
    )

    evaluate(
        "XGBoost",
        "Test",
        y_test[
            :,
            :sensor_count_for_xgboost,
        ],
        xgboost_predictions,
    )

    print()
    print("Sprint 3.3 completed.")
    print(
        f"XGBoost models trained: "
        f"{sensor_count_for_xgboost}",
    )


if __name__ == "__main__":
    main()
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.datasets.sequence_loader import (
    load_sequence_split,
)
from app.ml.evaluation.metrics import (
    mean_absolute_error,
    symmetric_mean_absolute_percentage_error,
    root_mean_squared_error,
)
from app.ml.evaluation.results import (
    EvaluationResult,
    save_evaluation_results,
)
from app.ml.models.persistence import PersistenceBaseline
from app.ml.models.xgboost_model import (
    TrafficXGBoostModel,
)


def evaluate(
    model_name: str,
    split: str,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> EvaluationResult:
    """Calculate evaluation metrics."""
    return EvaluationResult(
        model_name=model_name,
        split=split,
        mae=mean_absolute_error(
            actual,
            predicted,
        ),
        rmse=root_mean_squared_error(
            actual,
            predicted,
        ),
        smape=symmetric_mean_absolute_percentage_error(
            actual,
            predicted,
        ),
    )


def print_result(
    result: EvaluationResult,
) -> None:
    """Print an evaluation result."""
    print(
        f"{result.model_name} — {result.split}"
    )
    print("-" * 40)
    print(f"MAE:  {result.mae:.4f}")
    print(f"RMSE: {result.rmse:.4f}")
    print(f"MAPE: {result.smape:.2f}%")
    print()


def main() -> None:
    """Run the baseline forecasting experiment."""
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

    results_directory = (
        artifact_directory
        / "experiments"
    )

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

    print("Dataset")
    print("=======")
    print(f"Train:      {X_train.shape}")
    print(f"Validation: {X_validation.shape}")
    print(f"Test:       {X_test.shape}")
    print()

    results: list[EvaluationResult] = []

    # ---------------------------------------------------------
    # Persistence baseline
    # ---------------------------------------------------------

    persistence = PersistenceBaseline()

    validation_predictions = persistence.predict(
        X_validation,
    )

    validation_result = evaluate(
        model_name="persistence",
        split="validation",
        actual=y_validation,
        predicted=validation_predictions,
    )

    results.append(validation_result)
    print_result(validation_result)

    test_predictions = persistence.predict(
        X_test,
    )

    test_result = evaluate(
        model_name="persistence",
        split="test",
        actual=y_test,
        predicted=test_predictions,
    )

    results.append(test_result)
    print_result(test_result)

    # ---------------------------------------------------------
    # XGBoost
    # ---------------------------------------------------------

    model = TrafficXGBoostModel()

    print("Training XGBoost...")
    print()

    model.fit(
        X_train,
        y_train,
    )

    validation_predictions = model.predict(
        X_validation,
    )

    validation_result = evaluate(
        model_name="xgboost",
        split="validation",
        actual=y_validation,
        predicted=validation_predictions,
    )

    results.append(validation_result)
    print_result(validation_result)

    # Final test evaluation.
    test_predictions = model.predict(
        X_test,
    )

    test_result = evaluate(
        model_name="xgboost",
        split="test",
        actual=y_test,
        predicted=test_predictions,
    )

    results.append(test_result)
    print_result(test_result)

    # ---------------------------------------------------------
    # Save artifacts
    # ---------------------------------------------------------

    model_path = (
        artifact_directory
        / "xgboost_baseline.joblib"
    )

    model.save(model_path)

    results_path = (
        results_directory
        / "baseline_results.json"
    )

    save_evaluation_results(
        results,
        results_path,
    )

    print("Artifacts")
    print("=========")
    print(f"Model:   {model_path}")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
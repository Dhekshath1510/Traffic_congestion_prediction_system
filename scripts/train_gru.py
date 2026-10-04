from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from app.ml.models.gru_model import GRUConfig, TrafficGRUModel


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

SEED = 42

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

SEQUENCE_LENGTH = 12
NUM_SENSORS = 207

SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences"
)

RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR = RESULTS_DIR / "models"
PREDICTION_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"

MODEL_PATH = MODEL_DIR / "gru_residual_model.pt"

VALIDATION_PREDICTIONS_PATH = (
    PREDICTION_DIR / "gru_residual_validation.npy"
)

TEST_PREDICTIONS_PATH = (
    PREDICTION_DIR / "gru_residual_test.npy"
)

METRICS_PATH = (
    METRICS_DIR / "gru_residual_metrics.json"
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Configure deterministic random seeds where practical."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic execution is preferred for research reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_array(filename: str) -> np.ndarray:
    """Load one prepared NumPy sequence artifact."""

    path = SEQUENCE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required sequence file was not found: {path}"
        )

    array = np.load(path)

    if not np.isfinite(array).all():
        raise ValueError(
            f"Dataset contains NaN or infinite values: {path}"
        )

    return array.astype(np.float32, copy=False)


def load_datasets() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load train, validation and test sequence datasets."""

    X_train = load_array("X_train.npy")
    y_train = load_array("y_train.npy")

    X_validation = load_array("X_validation.npy")
    y_validation = load_array("y_validation.npy")

    X_test = load_array("X_test.npy")
    y_test = load_array("y_test.npy")

    validate_shapes(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )


def validate_shapes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Validate the expected multi-sensor sequence dimensions."""

    datasets = {
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "X_test": X_test,
        "y_test": y_test,
    }

    for name, array in datasets.items():
        if not np.isfinite(array).all():
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

    expected_X_shape_tail = (
        SEQUENCE_LENGTH,
        NUM_SENSORS,
    )

    expected_y_shape_tail = (NUM_SENSORS,)

    if X_train.shape[1:] != expected_X_shape_tail:
        raise ValueError(
            f"X_train must have shape "
            f"(samples, {SEQUENCE_LENGTH}, {NUM_SENSORS}). "
            f"Received {X_train.shape}."
        )

    if X_validation.shape[1:] != expected_X_shape_tail:
        raise ValueError(
            "X_validation has an unexpected shape: "
            f"{X_validation.shape}"
        )

    if X_test.shape[1:] != expected_X_shape_tail:
        raise ValueError(
            "X_test has an unexpected shape: "
            f"{X_test.shape}"
        )

    for name, X, y in (
        ("train", X_train, y_train),
        ("validation", X_validation, y_validation),
        ("test", X_test, y_test),
    ):
        if y.shape[1:] != expected_y_shape_tail:
            raise ValueError(
                f"y_{name} must have shape "
                f"(samples, {NUM_SENSORS}). "
                f"Received {y.shape}."
            )

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"{name} X/y sample counts do not match: "
                f"{X.shape[0]} != {y.shape[0]}"
            )


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def mean_absolute_error(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate global MAE across all samples and sensors."""

    return float(
        np.mean(
            np.abs(
                actual - predicted
            )
        )
    )


def root_mean_squared_error(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate global RMSE across all samples and sensors."""

    return float(
        np.sqrt(
            np.mean(
                np.square(
                    actual - predicted
                )
            )
        )
    )


def symmetric_mean_absolute_percentage_error(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Calculate sMAPE as a percentage.

    When both actual and predicted values are zero, the
    contribution is defined as zero.
    """

    denominator = (
        np.abs(actual)
        + np.abs(predicted)
    )

    numerator = (
        2.0
        * np.abs(actual - predicted)
    )

    ratio = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator != 0,
    )

    return float(
        np.mean(ratio) * 100.0
    )


def evaluate(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate the forecasting metrics."""

    return {
        "mae": mean_absolute_error(
            actual,
            predicted,
        ),
        "rmse": root_mean_squared_error(
            actual,
            predicted,
        ),
        "smape": symmetric_mean_absolute_percentage_error(
            actual,
            predicted,
        ),
    }


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------


def save_json(
    path: Path,
    payload: dict[str, object],
) -> None:
    """Save a JSON experiment artifact."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )


def save_model(
    model: TrafficGRUModel,
) -> None:
    """Save the trained PyTorch model and configuration."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": (
                model.model.state_dict()
            ),
            "config": {
                "input_size": (
                    model.config.input_size
                ),
                "hidden_size": (
                    model.config.hidden_size
                ),
                "num_layers": (
                    model.config.num_layers
                ),
                "dropout": (
                    model.config.dropout
                ),
                "output_size": (
                    model.config.output_size
                ),
            },
            "seed": SEED,
        },
        MODEL_PATH,
    )


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the Sprint 3.4 GRU forecasting experiment."""

    set_seed(SEED)

    print("=" * 64)
    print("Sprint 3.4 — GRU Traffic Forecasting")
    print("=" * 64)

    print("\nLoading sequence datasets...")

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = load_datasets()

    print("\nDataset")
    print("=======")
    print(f"Train:       {X_train.shape}")
    print(f"Train target:{y_train.shape}")
    print(f"Validation:  {X_validation.shape}")
    print(f"Val target:  {y_validation.shape}")
    print(f"Test:        {X_test.shape}")
    print(f"Test target: {y_test.shape}")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nTraining device: {device}")

    config = GRUConfig(
        input_size=NUM_SENSORS,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
        output_size=NUM_SENSORS,
    )

    print("\nGRU configuration")
    print("=================")
    print(f"Input sensors: {config.input_size}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"GRU layers: {config.num_layers}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")

    model = TrafficGRUModel(
        config=config,
        device=device,
    )

    print("\nTraining GRU...")
    print("=" * 64)

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
    )

    print("\nGenerating validation predictions...")

    validation_predictions = model.predict(
        X_validation
    )

    print("Generating test predictions...")

    test_predictions = model.predict(
        X_test
    )

    validation_metrics = evaluate(
        y_validation,
        validation_predictions,
    )

    test_metrics = evaluate(
        y_test,
        test_predictions,
    )

    print("\nGRU — Validation")
    print("=" * 64)
    print(
        f"MAE:   {validation_metrics['mae']:.4f}"
    )
    print(
        f"RMSE:  {validation_metrics['rmse']:.4f}"
    )
    print(
        f"sMAPE: {validation_metrics['smape']:.2f}%"
    )

    print("\nGRU — Test")
    print("=" * 64)
    print(
        f"MAE:   {test_metrics['mae']:.4f}"
    )
    print(
        f"RMSE:  {test_metrics['rmse']:.4f}"
    )
    print(
        f"sMAPE: {test_metrics['smape']:.2f}%"
    )

    # -----------------------------------------------------------------------
    # Save experiment artifacts
    # -----------------------------------------------------------------------

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        VALIDATION_PREDICTIONS_PATH,
        validation_predictions,
    )

    np.save(
        TEST_PREDICTIONS_PATH,
        test_predictions,
    )

    save_model(model)

    metrics_payload: dict[str, object] = {
        "experiment": "sprint_3.4c_residual_gru",
        "model_name": "gru_residual",
        "forecast_horizon_minutes": 5,
        "sequence_length": SEQUENCE_LENGTH,
        "num_sensors": NUM_SENSORS,
        "training_samples": int(
            X_train.shape[0]
        ),
        "validation_samples": int(
            X_validation.shape[0]
        ),
        "test_samples": int(
            X_test.shape[0]
        ),
        "configuration": {
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "device": device,
        },
        "validation": validation_metrics,
        "test": test_metrics,
    }

    save_json(
        METRICS_PATH,
        metrics_payload,
    )

    print("\nArtifacts saved")
    print("===============")
    print(f"Model:       {MODEL_PATH}")
    print(
        f"Validation:  "
        f"{VALIDATION_PREDICTIONS_PATH}"
    )
    print(
        f"Test:        "
        f"{TEST_PREDICTIONS_PATH}"
    )
    print(f"Metrics:     {METRICS_PATH}")

    print("\nSprint 3.4 completed.")


if __name__ == "__main__":
    main()
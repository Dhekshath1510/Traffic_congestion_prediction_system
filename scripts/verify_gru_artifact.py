from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


import numpy as np

from app.ml.models.gru_model import GRUConfig, TrafficGRUModel


MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "gru_residual_model.pt"
)

TEST_X_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences"
    / "X_test.npy"
)


def main() -> None:
    """Verify that the production GRU artifact loads and predicts."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"GRU checkpoint not found: {MODEL_PATH}"
        )

    if not TEST_X_PATH.exists():
        raise FileNotFoundError(
            f"Test sequence data not found: {TEST_X_PATH}"
        )

    X_test = np.load(TEST_X_PATH)

    if X_test.ndim != 3:
        raise RuntimeError(
            "Unexpected X_test dimensions: "
            f"{X_test.shape}"
        )

    if X_test.shape[1:] != (12, 207):
        raise RuntimeError(
            "Unexpected GRU input shape: "
            f"{X_test.shape[1:]}; "
            "expected (12, 207)"
        )

    model = TrafficGRUModel(
        GRUConfig(
            input_size=207,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            output_size=207,
        )
    )

    model.load_checkpoint(
        MODEL_PATH
    )

    predictions = model.predict(
        X_test[:8]
    )

    print(
        "GRU artifact verification successful."
    )
    print(
        f"Input shape:       {X_test[:8].shape}"
    )
    print(
        f"Prediction shape:  {predictions.shape}"
    )
    print(
        f"Prediction dtype:  {predictions.dtype}"
    )
    print(
        "Prediction finite: "
        f"{np.isfinite(predictions).all()}"
    )

    if predictions.shape != (8, 207):
        raise RuntimeError(
            "Unexpected prediction shape: "
            f"{predictions.shape}"
        )

    if predictions.dtype != np.float32:
        raise RuntimeError(
            "Unexpected prediction dtype: "
            f"{predictions.dtype}"
        )

    if not np.isfinite(predictions).all():
        raise RuntimeError(
            "GRU predictions contain non-finite values."
        )


if __name__ == "__main__":
    main()
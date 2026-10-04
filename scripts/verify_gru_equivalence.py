from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml.models.gru_model import GRUConfig, TrafficGRUModel


ORIGINAL_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "gru_residual_model.original.pt"
)

MIGRATED_MODEL_PATH = (
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
    """Verify migrated GRU predictions preserve original behavior."""

    if not ORIGINAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Original checkpoint not found: "
            f"{ORIGINAL_MODEL_PATH}"
        )

    if not MIGRATED_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Migrated checkpoint not found: "
            f"{MIGRATED_MODEL_PATH}"
        )

    X_test = np.load(TEST_X_PATH)

    X_sample = X_test[:8]

    # ------------------------------------------------------------------
    # Load migrated checkpoint through the production model loader.
    # ------------------------------------------------------------------

    migrated_model = TrafficGRUModel(
        GRUConfig(
            input_size=207,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            output_size=207,
        )
    )

    migrated_model.load_checkpoint(
        MIGRATED_MODEL_PATH
    )

    migrated_predictions = migrated_model.predict(
        X_sample
    )

    # ------------------------------------------------------------------
    # Load original weights manually.
    #
    # The original checkpoint did not contain normalization statistics,
    # so reconstruct them from the original training data exactly as
    # TrafficGRUModel.fit() did.
    # ------------------------------------------------------------------

    original_checkpoint = torch.load(
        ORIGINAL_MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    original_model = TrafficGRUModel(
        GRUConfig(
            input_size=207,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            output_size=207,
        )
    )

    original_model.model.load_state_dict(
        original_checkpoint[
            "model_state_dict"
        ]
    )

    X_train = np.load(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "sequences"
        / "X_train.npy"
    )

    mean = X_train.mean(
        axis=(0, 1),
        keepdims=True,
    )

    std = X_train.std(
        axis=(0, 1),
        keepdims=True,
    )

    std = np.where(
        std < 1e-6,
        1.0,
        std,
    )

    original_model._mean = mean.astype(
        np.float32
    )

    original_model._std = std.astype(
        np.float32
    )

    original_model._trained = True
    original_model.model.eval()

    original_predictions = original_model.predict(
        X_sample
    )

    # ------------------------------------------------------------------
    # Compare predictions.
    # ------------------------------------------------------------------

    absolute_difference = np.abs(
        original_predictions
        - migrated_predictions
    )

    max_difference = float(
        np.max(absolute_difference)
    )

    mean_difference = float(
        np.mean(absolute_difference)
    )

    print(
        "GRU prediction-equivalence verification"
    )
    print(
        f"Sample shape:       {X_sample.shape}"
    )
    print(
        f"Original shape:     {original_predictions.shape}"
    )
    print(
        f"Migrated shape:     {migrated_predictions.shape}"
    )
    print(
        f"Max absolute diff:  {max_difference:.12f}"
    )
    print(
        f"Mean absolute diff: {mean_difference:.12f}"
    )

    if not np.allclose(
        original_predictions,
        migrated_predictions,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise RuntimeError(
            "GRU prediction equivalence failed."
        )

    print(
        "GRU prediction equivalence verified successfully."
    )


if __name__ == "__main__":
    main()
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "gru_residual_model.pt"
)

TRAIN_X_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences"
    / "X_train.npy"
)


def main() -> None:
    """Add GRU normalization statistics to the existing checkpoint."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"GRU checkpoint not found: {MODEL_PATH}"
        )

    if not TRAIN_X_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_X_PATH}"
        )

    print(f"Loading checkpoint: {MODEL_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Invalid GRU checkpoint format."
        )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain model_state_dict."
        )

    if "mean" in checkpoint and "std" in checkpoint:
        print(
            "Checkpoint already contains normalization statistics."
        )
        return

    print(f"Loading training data: {TRAIN_X_PATH}")

    X_train = np.load(TRAIN_X_PATH)

    if X_train.ndim != 3:
        raise ValueError(
            "X_train must have shape "
            "(samples, timesteps, sensors). "
            f"Received: {X_train.shape}"
        )

    config = checkpoint.get("config")

    if not isinstance(config, dict):
        raise RuntimeError(
            "Checkpoint configuration is missing."
        )

    expected_sensors = int(
        config["input_size"]
    )

    if X_train.shape[-1] != expected_sensors:
        raise ValueError(
            "Training data sensor count does not match "
            "GRU input size: "
            f"{X_train.shape[-1]} != {expected_sensors}"
        )

    # These are exactly the normalization operations
    # used by TrafficGRUModel.fit().
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

    checkpoint["mean"] = mean.astype(
        np.float32
    )

    checkpoint["std"] = std.astype(
        np.float32
    )

    backup_path = MODEL_PATH.with_suffix(
        ".before_normalization.pt"
    )

    print(f"Creating backup: {backup_path}")

    torch.save(
        checkpoint,
        backup_path,
    )

    print(
        f"Writing migrated checkpoint: {MODEL_PATH}"
    )

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    print()
    print("Migration completed successfully.")
    print(f"mean shape: {checkpoint['mean'].shape}")
    print(f"std shape:  {checkpoint['std'].shape}")


if __name__ == "__main__":
    main()
from pathlib import Path

import numpy as np


def load_sequence_split(
    directory: Path,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load features and targets for a dataset split.

    Args:
        directory: Sequence directory.
        split_name: train, validation, or test.

    Returns:
        Tuple containing features and targets.
    """
    features_path = directory / f"X_{split_name}.npy"

    targets_path = directory / f"y_{split_name}.npy"

    if not features_path.exists():
        raise FileNotFoundError(
            f"Features not found: {features_path}",
        )

    if not targets_path.exists():
        raise FileNotFoundError(
            f"Targets not found: {targets_path}",
        )

    features = np.load(
        features_path,
    )

    targets = np.load(
        targets_path,
    )

    if len(features) != len(targets):
        raise ValueError(
            f"Feature/target length mismatch for "
            f"{split_name}: "
            f"{len(features)} != {len(targets)}",
        )

    return features, targets

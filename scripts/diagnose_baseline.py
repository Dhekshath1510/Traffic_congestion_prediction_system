from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.datasets.sequence_loader import (
    load_sequence_split,
)


def describe(
    name: str,
    targets: np.ndarray,
) -> None:
    """Print target distribution statistics."""
    zero_count = np.sum(targets <= 1e-6)

    print(f"\n{name}")
    print("=" * len(name))
    print(f"Samples:       {len(targets)}")
    print(f"Minimum:       {np.min(targets):.4f}")
    print(f"Maximum:       {np.max(targets):.4f}")
    print(f"Mean:          {np.mean(targets):.4f}")
    print(f"Median:        {np.median(targets):.4f}")
    print(
        f"Zero/near-zero: {zero_count} "
        f"({zero_count / len(targets) * 100:.2f}%)",
    )
    print(
        f"Std:            {np.std(targets):.4f}",
    )


def main() -> None:
    """Diagnose the baseline target distribution."""
    sequence_directory = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "sequences"
    )

    _, y_train = load_sequence_split(
        sequence_directory,
        "train",
    )

    _, y_validation = load_sequence_split(
        sequence_directory,
        "validation",
    )

    _, y_test = load_sequence_split(
        sequence_directory,
        "test",
    )

    describe("Training target", y_train)
    describe("Validation target", y_validation)
    describe("Test target", y_test)


if __name__ == "__main__":
    main()
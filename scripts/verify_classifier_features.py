from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from app.ml.classification.feature_builder import (
    BASE_FEATURE_NAMES,
    CLASSIFIER_FEATURE_NAMES,
    build_base_features,
)


SEQUENCES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences"
    / "X_test.npy"
)

TIMESTAMPS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "congestion"
    / "timestamps_test.npy"
)

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "congestion"
    / "features_test.csv"
)


def main() -> None:
    """Verify production features against persisted training features."""

    X_test = np.load(
        SEQUENCES_PATH,
        allow_pickle=False,
    )

    timestamps = np.load(
        TIMESTAMPS_PATH,
        allow_pickle=False,
    )

    persisted = pd.read_csv(
        FEATURES_PATH
    )

    if X_test.ndim != 3:
        raise RuntimeError(
            f"Unexpected X_test shape: {X_test.shape}"
        )

    if X_test.shape[1:] != (12, 207):
        raise RuntimeError(
            "Unexpected sequence shape: "
            f"{X_test.shape[1:]}"
        )

    if len(timestamps) != len(X_test):
        raise RuntimeError(
            "Timestamp/sample mismatch: "
            f"{len(timestamps)} != {len(X_test)}"
        )

    expected_columns = [
        "timestamp",
        *BASE_FEATURE_NAMES,
    ]

    if list(persisted.columns) != expected_columns:
        raise RuntimeError(
            "Persisted feature columns do not match "
            "the expected 24-feature schema."
        )

    # Test several samples across the test set rather than
    # checking only the first row.
    test_indices = [
        0,
        1,
        len(X_test) // 2,
        len(X_test) - 1,
    ]

    max_difference = 0.0

    for index in test_indices:
        timestamp = pd.Timestamp(
            timestamps[index]
        ).to_pydatetime()

        generated = build_base_features(
            sequence=X_test[index],
            timestamp=timestamp,
        )

        expected = persisted.loc[
            index,
            list(BASE_FEATURE_NAMES),
        ].to_numpy(
            dtype=np.float32
        )

        difference = np.abs(
            generated - expected
        )

        sample_max_difference = float(
            np.max(difference)
        )

        max_difference = max(
            max_difference,
            sample_max_difference,
        )

        print(
            f"Sample {index}: "
            f"max absolute difference = "
            f"{sample_max_difference:.10f}"
        )

        if not np.allclose(
            generated,
            expected,
            rtol=1e-5,
            atol=1e-5,
        ):
            raise RuntimeError(
                "Feature parity failed for "
                f"sample {index}."
            )

    # Verify final classifier feature schema.
    if len(CLASSIFIER_FEATURE_NAMES) != 25:
        raise RuntimeError(
            "Expected exactly 25 classifier features."
        )

    print()
    print(
        "Classifier base-feature parity verified "
        "successfully."
    )
    print(
        f"Maximum absolute difference: "
        f"{max_difference:.10f}"
    )
    print(
        f"Base feature count: "
        f"{len(BASE_FEATURE_NAMES)}"
    )
    print(
        f"Classifier feature count: "
        f"{len(CLASSIFIER_FEATURE_NAMES)}"
    )


if __name__ == "__main__":
    main()
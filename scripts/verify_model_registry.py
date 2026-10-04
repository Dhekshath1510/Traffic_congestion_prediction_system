from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Project import setup
# ---------------------------------------------------------------------------
# The application package lives under backend/app.
# When this script is executed from the project root, Python does not
# automatically include "backend" on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from app.ml.runtime.artifacts import ModelArtifactPaths
from app.ml.runtime.model_registry import ModelRegistry


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEQUENCE_LENGTH = 12
SENSOR_COUNT = 207
TEST_BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def verify_gru(registry: ModelRegistry) -> None:
    """Verify that the GRU artifact loads and performs inference."""

    gru = registry.gru

    if not gru._trained:
        raise AssertionError(
            "GRU model is loaded but is not marked as trained."
        )

    if gru.config.input_size != SENSOR_COUNT:
        raise AssertionError(
            "Unexpected GRU input size: "
            f"{gru.config.input_size}. "
            f"Expected {SENSOR_COUNT}."
        )

    if gru.config.output_size != SENSOR_COUNT:
        raise AssertionError(
            "Unexpected GRU output size: "
            f"{gru.config.output_size}. "
            f"Expected {SENSOR_COUNT}."
        )

    if gru._mean.shape != (1, 1, SENSOR_COUNT):
        raise AssertionError(
            "Unexpected GRU normalization mean shape: "
            f"{gru._mean.shape}."
        )

    if gru._std.shape != (1, 1, SENSOR_COUNT):
        raise AssertionError(
            "Unexpected GRU normalization std shape: "
            f"{gru._std.shape}."
        )

    if not np.isfinite(gru._mean).all():
        raise AssertionError(
            "GRU normalization mean contains "
            "NaN or infinite values."
        )

    if not np.isfinite(gru._std).all():
        raise AssertionError(
            "GRU normalization std contains "
            "NaN or infinite values."
        )

    if np.any(gru._std <= 0):
        raise AssertionError(
            "GRU normalization std must be strictly positive."
        )

    # Deterministic synthetic inference input.
    rng = np.random.default_rng(42)

    X = rng.uniform(
        low=0.0,
        high=70.0,
        size=(
            TEST_BATCH_SIZE,
            SEQUENCE_LENGTH,
            SENSOR_COUNT,
        ),
    ).astype(np.float32)

    predictions = gru.predict(X)

    expected_shape = (
        TEST_BATCH_SIZE,
        SENSOR_COUNT,
    )

    if predictions.shape != expected_shape:
        raise AssertionError(
            "Unexpected GRU prediction shape: "
            f"{predictions.shape}. "
            f"Expected {expected_shape}."
        )

    if predictions.dtype != np.float32:
        raise AssertionError(
            "Unexpected GRU prediction dtype: "
            f"{predictions.dtype}. Expected float32."
        )

    if not np.isfinite(predictions).all():
        raise AssertionError(
            "GRU predictions contain NaN or infinite values."
        )

    print("GRU model verification:")
    print(f"  trained:              {gru._trained}")
    print(f"  input_size:           {gru.config.input_size}")
    print(f"  output_size:          {gru.config.output_size}")
    print(f"  normalization mean:   {gru._mean.shape}")
    print(f"  normalization std:    {gru._std.shape}")
    print(f"  input shape:          {X.shape}")
    print(f"  prediction shape:     {predictions.shape}")
    print(f"  prediction dtype:     {predictions.dtype}")
    print("  prediction finite:    True")
    print()


def verify_classifier(registry: ModelRegistry) -> None:
    """Verify classifier artifact loading and inference."""

    classifier = registry.classifier
    model = classifier.model

    if model.n_features_in_ != 25:
        raise AssertionError(
            "Unexpected classifier feature count: "
            f"{model.n_features_in_}. Expected 25."
        )

    expected_classes = np.array([0, 1, 2])

    if not np.array_equal(
        model.classes_,
        expected_classes,
    ):
        raise AssertionError(
            "Unexpected classifier classes: "
            f"{model.classes_}. "
            f"Expected {expected_classes}."
        )

    feature_importance = classifier.get_feature_importance()

    if feature_importance.shape != (25,):
        raise AssertionError(
            "Unexpected feature-importance shape: "
            f"{feature_importance.shape}. Expected (25,)."
        )

    if not np.isfinite(feature_importance).all():
        raise AssertionError(
            "Classifier feature importance contains "
            "NaN or infinite values."
        )

    # Deterministic synthetic classifier input.
    rng = np.random.default_rng(42)

    X = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(
            TEST_BATCH_SIZE,
            25,
        ),
    ).astype(np.float32)

    predictions = classifier.predict(X)
    probabilities = classifier.predict_proba(X)

    expected_prediction_shape = (TEST_BATCH_SIZE,)
    expected_probability_shape = (
        TEST_BATCH_SIZE,
        3,
    )

    if predictions.shape != expected_prediction_shape:
        raise AssertionError(
            "Unexpected classifier prediction shape: "
            f"{predictions.shape}. "
            f"Expected {expected_prediction_shape}."
        )

    if probabilities.shape != expected_probability_shape:
        raise AssertionError(
            "Unexpected classifier probability shape: "
            f"{probabilities.shape}. "
            f"Expected {expected_probability_shape}."
        )

    if predictions.dtype != np.int8:
        raise AssertionError(
            "Unexpected classifier prediction dtype: "
            f"{predictions.dtype}. Expected int8."
        )

    if probabilities.dtype != np.float32:
        raise AssertionError(
            "Unexpected classifier probability dtype: "
            f"{probabilities.dtype}. Expected float32."
        )

    if not np.isfinite(probabilities).all():
        raise AssertionError(
            "Classifier probabilities contain "
            "NaN or infinite values."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-5,
    ):
        raise AssertionError(
            "Classifier probabilities do not sum to 1."
        )

    if not np.isin(
        predictions,
        expected_classes,
    ).all():
        raise AssertionError(
            "Classifier produced an invalid class ID."
        )

    print("Congestion classifier verification:")
    print(f"  model type:           {type(model).__name__}")
    print(f"  n_features_in_:       {model.n_features_in_}")
    print(f"  classes_:             {model.classes_}")
    print(f"  feature importance:   {feature_importance.shape}")
    print(f"  input shape:          {X.shape}")
    print(f"  prediction shape:     {predictions.shape}")
    print(f"  probability shape:    {probabilities.shape}")
    print(f"  prediction dtype:     {predictions.dtype}")
    print(f"  probability dtype:    {probabilities.dtype}")
    print("  probabilities valid:  True")
    print()


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------

def main() -> None:
    """Load the production artifacts and verify real inference."""

    print("Model Registry Verification")
    print("=" * 32)
    print()

    artifact_paths = ModelArtifactPaths.from_project_root(
        PROJECT_ROOT,
    )

    print("Artifact paths:")
    print(f"  GRU:         {artifact_paths.gru_checkpoint}")
    print(
        "  Classifier: "
        f"{artifact_paths.congestion_classifier}"
    )
    print()

    artifact_paths.validate()

    registry = ModelRegistry(
        artifact_paths,
    )

    registry.load()

    print("Model registry loaded successfully.")
    print()

    verify_gru(registry)
    verify_classifier(registry)

    print("=" * 32)
    print("MODEL REGISTRY VERIFICATION SUCCESSFUL")
    print("=" * 32)


if __name__ == "__main__":
    main()
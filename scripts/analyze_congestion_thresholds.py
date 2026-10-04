from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences"
)

RESULTS_DIR = PROJECT_ROOT / "results"

CLASSIFICATION_DIR = (
    RESULTS_DIR / "classification"
)

FIGURES_DIR = (
    RESULTS_DIR / "figures"
)

CLASSIFICATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_TARGET = SEQUENCE_DIR / "y_train.npy"
VALIDATION_TARGET = SEQUENCE_DIR / "y_validation.npy"
TEST_TARGET = SEQUENCE_DIR / "y_test.npy"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_target(path: Path) -> np.ndarray:
    """Load and validate a traffic-speed target array."""

    if not path.exists():
        raise FileNotFoundError(
            f"Target file not found: {path}"
        )

    values = np.load(path)

    if values.ndim != 2:
        raise ValueError(
            f"Expected 2-D target array, got {values.shape}"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            f"Target contains NaN or infinite values: {path}"
        )

    return values.astype(
        np.float32,
        copy=False,
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def describe_distribution(
    name: str,
    values: np.ndarray,
) -> None:
    """Print descriptive statistics."""

    flattened = values.reshape(-1)

    print(f"\n{name}")
    print("=" * len(name))

    print(f"Samples:       {len(flattened):,}")
    print(f"Minimum:       {np.min(flattened):.4f}")
    print(f"Maximum:       {np.max(flattened):.4f}")
    print(f"Mean:          {np.mean(flattened):.4f}")
    print(f"Median:        {np.median(flattened):.4f}")
    print(f"Std:           {np.std(flattened):.4f}")

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    print("\nPercentiles")

    for percentile in percentiles:
        value = np.percentile(
            flattened,
            percentile,
        )

        print(
            f"{percentile:>2}th:          "
            f"{value:.4f}"
        )

    zero_count = np.count_nonzero(
        flattened <= 1.0
    )

    print(
        "\nZero/near-zero "
        f"(<= 1): {zero_count:,} "
        f"({100.0 * zero_count / len(flattened):.2f}%)"
    )


def create_distribution_dataframe(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> pd.DataFrame:
    """Create a tidy dataframe for distribution analysis."""

    frames: list[pd.DataFrame] = []

    for name, values in (
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ):
        flattened = values.reshape(-1)

        frames.append(
            pd.DataFrame(
                {
                    "split": name,
                    "speed": flattened,
                }
            )
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Candidate threshold analysis
# ---------------------------------------------------------------------------


def evaluate_candidate_thresholds(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate candidate speed thresholds.

    These are candidate thresholds only. They are NOT automatically
    accepted as the final congestion definition.
    """

    candidates = [
        (20.0, 40.0),
        (25.0, 45.0),
        (30.0, 50.0),
        (35.0, 50.0),
        (40.0, 55.0),
        (45.0, 55.0),
    ]

    rows: list[dict[str, float | str]] = []

    test_values = dataframe.loc[
        dataframe["split"] == "Test",
        "speed",
    ].to_numpy()

    for low_moderate, moderate_severe in candidates:
        labels = np.select(
            [
                test_values >= moderate_severe,
                test_values >= low_moderate,
            ],
            [
                "LOW",
                "MODERATE",
            ],
            default="SEVERE",
        )

        counts = pd.Series(
            labels
        ).value_counts()

        total = len(labels)

        rows.append(
            {
                "low_moderate_threshold": (
                    low_moderate
                ),
                "moderate_severe_threshold": (
                    moderate_severe
                ),
                "low_percent": (
                    100.0
                    * counts.get("LOW", 0)
                    / total
                ),
                "moderate_percent": (
                    100.0
                    * counts.get("MODERATE", 0)
                    / total
                ),
                "severe_percent": (
                    100.0
                    * counts.get("SEVERE", 0)
                    / total
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_speed_distribution(
    dataframe: pd.DataFrame,
) -> None:
    """Generate the traffic-speed distribution figure."""

    # Sample only for plotting to keep the figure lightweight.
    max_points = 100_000

    if len(dataframe) > max_points:
        plot_data = dataframe.sample(
            n=max_points,
            random_state=42,
        )
    else:
        plot_data = dataframe

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for split in (
        "Train",
        "Validation",
        "Test",
    ):
        values = plot_data.loc[
            plot_data["split"] == split,
            "speed",
        ]

        ax.hist(
            values,
            bins=50,
            alpha=0.35,
            label=split,
        )

    ax.set_title(
        "Traffic Speed Distribution Across Dataset Splits"
    )

    ax.set_xlabel(
        "Traffic speed"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.legend()

    fig.tight_layout()

    path = (
        FIGURES_DIR
        / "traffic_speed_distribution.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved: {path}")


def plot_candidate_class_balance(
    candidates: pd.DataFrame,
) -> None:
    """Visualize class proportions for candidate thresholds."""

    x = np.arange(
        len(candidates)
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        x - width,
        candidates["low_percent"],
        width,
        label="LOW",
    )

    ax.bar(
        x,
        candidates["moderate_percent"],
        width,
        label="MODERATE",
    )

    ax.bar(
        x + width,
        candidates["severe_percent"],
        width,
        label="SEVERE",
    )

    labels = [
        (
            f"{row.low_moderate_threshold:.0f}/"
            f"{row.moderate_severe_threshold:.0f}"
        )
        for row in candidates.itertuples()
    ]

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_xlabel(
        "Candidate thresholds "
        "(LOW/MODERATE)"
    )

    ax.set_ylabel(
        "Samples (%)"
    )

    ax.set_title(
        "Congestion Class Distribution Under Candidate Thresholds"
    )

    ax.legend()

    fig.tight_layout()

    path = (
        FIGURES_DIR
        / "candidate_congestion_class_balance.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Sprint 4.1 congestion-threshold analysis."""

    print("=" * 64)
    print("Sprint 4.1 — Congestion Threshold Analysis")
    print("=" * 64)

    train = load_target(
        TRAIN_TARGET
    )

    validation = load_target(
        VALIDATION_TARGET
    )

    test = load_target(
        TEST_TARGET
    )

    print("\nDataset shapes")
    print("==============")
    print(f"Train:      {train.shape}")
    print(f"Validation: {validation.shape}")
    print(f"Test:       {test.shape}")

    describe_distribution(
        "Training target",
        train,
    )

    describe_distribution(
        "Validation target",
        validation,
    )

    describe_distribution(
        "Test target",
        test,
    )

    dataframe = create_distribution_dataframe(
        train,
        validation,
        test,
    )

    distribution_path = (
        CLASSIFICATION_DIR
        / "traffic_speed_distribution.csv"
    )

    dataframe.to_csv(
        distribution_path,
        index=False,
    )

    print(
        f"\nSaved distribution data: "
        f"{distribution_path}"
    )

    candidates = evaluate_candidate_thresholds(
        dataframe
    )

    candidate_path = (
        CLASSIFICATION_DIR
        / "candidate_thresholds.csv"
    )

    candidates.to_csv(
        candidate_path,
        index=False,
    )

    print(
        f"Saved candidate analysis: "
        f"{candidate_path}"
    )

    print("\nCandidate thresholds")
    print("====================")

    print(
        candidates.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.2f}"
            ),
        )
    )

    plot_speed_distribution(
        dataframe
    )

    plot_candidate_class_balance(
        candidates
    )

    print("\nSprint 4.1 completed.")


if __name__ == "__main__":
    main()
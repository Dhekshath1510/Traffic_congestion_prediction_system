import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.datasets.sequences import (
    SequenceConfig,
    TrafficSequenceGenerator,
)


def generate_split(
    dataframe: pd.DataFrame,
    split_name: str,
    output_directory: Path,
    generator: TrafficSequenceGenerator,
) -> None:
    """Generate and save sequences for one dataset split."""
    dataset = generator.generate(dataframe)

    np.save(
        output_directory / f"X_{split_name}.npy",
        dataset.features,
    )

    np.save(
        output_directory / f"y_{split_name}.npy",
        dataset.targets,
    )

    np.save(
        output_directory / f"timestamps_{split_name}.npy",
        dataset.timestamps,
    )

    print(
        f"{split_name}: "
        f"X={dataset.features.shape}, "
        f"y={dataset.targets.shape}",
    )


def main() -> None:
    """Generate supervised sequences for all dataset splits."""
    processed_directory = (
        PROJECT_ROOT
        / "data"
        / "processed"
    )

    sequence_directory = (
        processed_directory
        / "sequences"
    )

    sequence_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    generator = TrafficSequenceGenerator(
        SequenceConfig(
            history_steps=12,
            horizon_steps=3,
        ),
    )

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        dataframe = pd.read_csv(
            processed_directory
            / f"{split_name}.csv",
        )

        generate_split(
            dataframe=dataframe,
            split_name=split_name,
            output_directory=sequence_directory,
            generator=generator,
        )

    print("\nSequence generation completed.")


if __name__ == "__main__":
    main()
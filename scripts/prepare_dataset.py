import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.datasets.loader import TrafficDatasetLoader
from app.ml.datasets.preprocessor import (
    TrafficDatasetPreprocessor,
)

def main() -> None:
    """Prepare the METR-LA dataset."""
    raw_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "METR-LA.csv"
    )

    processed_directory = (
        PROJECT_ROOT
        / "data"
        / "processed"
    )

    processed_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    loader = TrafficDatasetLoader()

    dataframe = loader.load_csv(raw_path)

    preprocessor = TrafficDatasetPreprocessor()

    splits = preprocessor.split(dataframe)

    train = preprocessor.fit_transform(
        splits.train,
    )

    validation = preprocessor.transform(
        splits.validation,
    )

    test = preprocessor.transform(
        splits.test,
    )

    train.to_csv(
        processed_directory / "train.csv",
        index=False,
    )

    validation.to_csv(
        processed_directory / "validation.csv",
        index=False,
    )

    test.to_csv(
        processed_directory / "test.csv",
        index=False,
    )

    print("Dataset preparation completed.")
    print(f"Train rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")
    print(f"Test rows: {len(test)}")


if __name__ == "__main__":
    main()
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.datasets.config import DatasetConfig
from app.ml.datasets.loader import TrafficDatasetLoader
from app.ml.datasets.profiler import TrafficDatasetProfiler
from app.ml.datasets.report import format_profile_report


def main() -> None:
    """Profile the METR-LA dataset."""
    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "METR-LA.csv"
    )

    config = DatasetConfig()
    loader = TrafficDatasetLoader()
    dataframe = loader.load_csv(dataset_path)
    profiler = TrafficDatasetProfiler()
    profile = profiler.profile(
        dataframe=dataframe,
        expected_interval_minutes=(
            config.expected_interval_minutes
        ),
    )

    print(format_profile_report(profile))


if __name__ == "__main__":
    main()
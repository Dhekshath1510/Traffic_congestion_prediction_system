from pathlib import Path
import sys
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_ROOT))


def main() -> None:
    """Analyze traffic-speed distributions across sensors."""
    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "METR-LA.csv"
    )

    dataframe = pd.read_csv(
        dataset_path,
    )

    timestamp_column = "timestamp"

    if timestamp_column not in dataframe.columns:
        first_column = dataframe.columns[0]

        dataframe = dataframe.rename(
            columns={
                first_column: timestamp_column,
            },
        )

    sensor_columns = [
        column
        for column in dataframe.columns
        if column != timestamp_column
    ]

    values = dataframe[sensor_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    statistics = pd.DataFrame(
        {
            "mean": values.mean(),
            "median": values.median(),
            "std": values.std(),
            "minimum": values.min(),
            "maximum": values.max(),
            "zero_percentage": (
                (values <= 1e-6).mean()
                * 100.0
            ),
            "missing_percentage": (
                values.isna().mean()
                * 100.0
            ),
        }
    )

    statistics = statistics.sort_values(
        "zero_percentage",
        ascending=False,
    )

    print("\nSensor Statistics")
    print("=================")

    print(
        statistics.to_string(
            float_format=lambda value: f"{value:.2f}",
        ),
    )

    print("\nNetwork Summary")
    print("================")

    print(
        f"Number of sensors: {len(sensor_columns)}",
    )

    print(
        f"Average sensor mean: "
        f"{statistics['mean'].mean():.2f}",
    )

    print(
        f"Average zero percentage: "
        f"{statistics['zero_percentage'].mean():.2f}%",
    )

    print(
        f"Maximum zero percentage: "
        f"{statistics['zero_percentage'].max():.2f}%",
    )

    print(
        f"Minimum zero percentage: "
        f"{statistics['zero_percentage'].min():.2f}%",
    )


if __name__ == "__main__":
    main()
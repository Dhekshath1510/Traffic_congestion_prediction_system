from app.ml.datasets.profiler import DatasetProfile


def format_profile_report(
    profile: DatasetProfile,
) -> str:
    """Format a dataset profile for terminal output.

    Args:
        profile: Dataset profile.

    Returns:
        Human-readable profile report.
    """
    statistics = profile.sensor_statistics

    return (
        "Traffic Dataset Profile\n"
        "=======================\n"
        f"Rows: {profile.row_count}\n"
        f"Sensors: {profile.sensor_count}\n"
        f"Duplicate rows: {profile.duplicate_rows}\n"
        "\n"
        "Temporal Coverage\n"
        "-----------------\n"
        f"Start: {profile.start_timestamp}\n"
        f"End: {profile.end_timestamp}\n"
        f"Expected interval: "
        f"{profile.expected_interval_minutes} minutes\n"
        f"Observed median interval: "
        f"{profile.median_interval_minutes} minutes\n"
        f"Missing timestamps: "
        f"{profile.missing_timestamp_count}\n"
        "\n"
        "Data Quality\n"
        "------------\n"
        f"Missing sensor values: "
        f"{profile.missing_value_count}\n"
        f"Invalid negative values: "
        f"{profile.invalid_value_count}\n"
        "\n"
        "Traffic Speed Statistics\n"
        "------------------------\n"
        f"Minimum: {statistics.get('minimum')}\n"
        f"Maximum: {statistics.get('maximum')}\n"
        f"Mean: {statistics.get('mean')}\n"
        f"Median: {statistics.get('median')}\n"
    )

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation metrics for one forecasting model."""

    model_name: str
    split: str
    mae: float
    rmse: float
    smape: float

    def to_dict(self) -> dict[str, object]:
        """Convert the result to a serializable dictionary."""
        return asdict(self)


def save_evaluation_results(
    results: list[EvaluationResult],
    path: Path,
) -> None:
    """Save evaluation results as JSON."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = [result.to_dict() for result in results]

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

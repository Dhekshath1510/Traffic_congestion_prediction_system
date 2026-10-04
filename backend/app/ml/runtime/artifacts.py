from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelArtifactPaths:
    """Filesystem locations of production ML artifacts."""

    gru_checkpoint: Path
    congestion_classifier: Path

    @classmethod
    def from_project_root(
        cls,
        project_root: Path,
    ) -> "ModelArtifactPaths":
        """Build artifact paths relative to the project root."""

        models_dir = project_root / "results" / "models"

        return cls(
            gru_checkpoint=models_dir / "gru_residual_model.pt",
            congestion_classifier=models_dir
            / "congestion_classifier.joblib",
        )

    def validate(self) -> None:
        """Validate that all required model artifacts exist."""

        missing: list[str] = []

        if not self.gru_checkpoint.is_file():
            missing.append(str(self.gru_checkpoint))

        if not self.congestion_classifier.is_file():
            missing.append(str(self.congestion_classifier))

        if missing:
            raise FileNotFoundError(
                "Required ML model artifacts are missing:\n"
                + "\n".join(missing)
            )
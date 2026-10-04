from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam


@dataclass(frozen=True)
class GRUConfig:
    """Configuration for the compact residual GRU."""

    input_size: int = 207
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    output_size: int = 207


class TrafficGRU(nn.Module):
    """Compact GRU that predicts a correction to the latest observation."""

    def __init__(self, config: GRUConfig) -> None:
        super().__init__()

        self.config = config

        self.gru = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=(
                config.dropout
                if config.num_layers > 1
                else 0.0
            ),
        )

        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, config.output_size),
        )

    
    def forward(self, x: Tensor) -> Tensor:
        """Return the predicted residual for each sensor."""

        if x.ndim != 3:
            raise ValueError(
                "Expected input with shape "
                "(batch, sequence_length, sensors). "
                f"Received {tuple(x.shape)}."
            )

        if x.shape[-1] != self.config.input_size:
            raise ValueError(
                f"Expected {self.config.input_size} sensors, "
                f"received {x.shape[-1]}."
            )

        output, _ = self.gru(x)

        final_hidden = output[:, -1, :]

        return self.head(final_hidden)


class TrafficGRUModel:
    """
    Production-facing residual GRU wrapper.

    The model learns:

        residual = target - latest_observation

    and reconstructs:

        prediction = latest_observation + residual
    """

    def __init__(
        self,
        config: GRUConfig | None = None,
        *,
        device: str | None = None,
    ) -> None:
        self.config = config or GRUConfig()

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        self.model = TrafficGRU(
            self.config
        ).to(self.device)

        self._trained = False

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
    ) -> None:
        """Load a production GRU checkpoint."""

        path = Path(checkpoint_path)

        if not path.exists():
            raise FileNotFoundError(
                f"GRU checkpoint not found: {path}"
            )

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                "Invalid GRU checkpoint format."
            )

        state_dict = checkpoint.get(
            "model_state_dict"
        )

        if state_dict is None:
            raise RuntimeError(
                "GRU checkpoint is missing "
                "'model_state_dict'."
            )

        checkpoint_config = checkpoint.get(
            "config"
        )

        if checkpoint_config is not None:
            expected = {
                "input_size": self.config.input_size,
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "output_size": self.config.output_size,
            }

            for key, expected_value in expected.items():
                actual_value = checkpoint_config.get(key)

                if actual_value != expected_value:
                    raise RuntimeError(
                        "GRU checkpoint configuration mismatch "
                        f"for '{key}': "
                        f"{actual_value} != {expected_value}"
                    )

        mean = checkpoint.get("mean")
        std = checkpoint.get("std")

        if mean is None or std is None:
            raise RuntimeError(
                "GRU checkpoint does not contain "
                "normalization statistics. "
                "Run scripts/migrate_gru_checkpoint.py first."
            )

        mean_array = np.asarray(
            mean,
            dtype=np.float32,
        )

        std_array = np.asarray(
            std,
            dtype=np.float32,
        )

        expected_shape = (
            1,
            1,
            self.config.input_size,
        )

        if mean_array.shape != expected_shape:
            raise RuntimeError(
                "Invalid GRU mean shape: "
                f"{mean_array.shape}; "
                f"expected {expected_shape}"
            )

        if std_array.shape != expected_shape:
            raise RuntimeError(
                "Invalid GRU std shape: "
                f"{std_array.shape}; "
                f"expected {expected_shape}"
            )

        if not np.isfinite(mean_array).all():
            raise RuntimeError(
                "GRU normalization mean contains "
                "non-finite values."
            )

        if not np.isfinite(std_array).all():
            raise RuntimeError(
                "GRU normalization std contains "
                "non-finite values."
            )

        if np.any(std_array <= 0.0):
            raise RuntimeError(
                "GRU normalization std must be positive."
            )

        self.model.load_state_dict(
            state_dict
        )

        self._mean = mean_array
        self._std = std_array

        self.model.eval()
        self._trained = True

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
    ) -> None:
        """Train the residual GRU using train-only normalization."""

        self._validate_training_data(
            X_train,
            y_train,
        )

        X_mean = X_train.mean(
            axis=(0, 1),
            keepdims=True,
        )

        X_std = X_train.std(
            axis=(0, 1),
            keepdims=True,
        )

        X_std = np.where(
            X_std < 1e-6,
            1.0,
            X_std,
        )

        self._mean = X_mean.astype(
            np.float32
        )

        self._std = X_std.astype(
            np.float32
        )

        X_normalized = (
            X_train - self._mean
        ) / self._std

        latest_observation = X_train[:, -1, :]

        residual_target = (
            y_train - latest_observation
        )

        X_tensor = torch.as_tensor(
            X_normalized,
            dtype=torch.float32,
        )

        residual_tensor = torch.as_tensor(
            residual_target,
            dtype=torch.float32,
        )

        dataset = torch.utils.data.TensorDataset(
            X_tensor,
            residual_tensor,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        optimizer: Adam = Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

        criterion = nn.HuberLoss(
            delta=1.0
        )

        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch_x, batch_residual in loader:
                batch_x = batch_x.to(
                    self.device
                )

                batch_residual = batch_residual.to(
                    self.device
                )

                optimizer.zero_grad()

                predicted_residual = self.model(
                    batch_x
                )

                loss = criterion(
                    predicted_residual,
                    batch_residual,
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

                epoch_loss += (
                    loss.item()
                    * batch_x.size(0)
                )

            average_loss = (
                epoch_loss / len(dataset)
            )

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"- loss: {average_loss:.6f}"
            )

        self._trained = True

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Predict the next traffic state."""

        if not self._trained:
            raise RuntimeError(
                "The GRU model must be trained "
                "before prediction."
            )

        if X.ndim != 3:
            raise ValueError(
                "Expected X with shape "
                "(samples, sequence_length, sensors)."
            )

        if X.shape[-1] != self.config.input_size:
            raise ValueError(
                f"Expected {self.config.input_size} sensors, "
                f"received {X.shape[-1]}."
            )

        X_normalized = (
            X - self._mean
        ) / self._std

        X_tensor = torch.as_tensor(
            X_normalized,
            dtype=torch.float32,
            device=self.device,
        )

        latest_observation = X[:, -1, :]

        predictions: list[np.ndarray] = []

        self.model.eval()

        with torch.no_grad():
            for batch in torch.split(
                X_tensor,
                256,
                dim=0,
            ):
                predicted_residual = self.model(
                    batch
                )

                predictions.append(
                    predicted_residual.cpu().numpy()
                )

        residual = np.concatenate(
            predictions,
            axis=0,
        )

        return (
            latest_observation
            + residual
        ).astype(np.float32)

    def _validate_training_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """Validate training array dimensions."""

        if X_train.ndim != 3:
            raise ValueError(
                "X_train must have shape "
                "(samples, sequence_length, sensors)."
            )

        if y_train.ndim != 2:
            raise ValueError(
                "y_train must have shape "
                "(samples, sensors)."
            )

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                "X_train and y_train must contain "
                "the same number of samples."
            )

        if X_train.shape[-1] != self.config.input_size:
            raise ValueError(
                f"Expected {self.config.input_size} sensors."
            )

        if y_train.shape[-1] != self.config.output_size:
            raise ValueError(
                f"Expected {self.config.output_size} target sensors."
            )
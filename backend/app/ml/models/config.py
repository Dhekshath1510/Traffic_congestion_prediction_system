from dataclasses import dataclass


@dataclass(frozen=True)
class XGBoostConfig:
    """Configuration for the XGBoost baseline."""

    n_estimators: int = 150
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42

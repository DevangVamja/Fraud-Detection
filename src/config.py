"""Configuration objects for the fraud detection training pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DataConfig:
    """Configuration describing how to load and prepare the dataset."""

    csv_path: Path
    target_column: str = "isFraud"
    drop_columns: List[str] | None = None


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the model training routine."""

    test_size: float = 0.2
    random_state: int = 42
    model_output_path: Path = Path("models/fraud_model.joblib")
    metrics_output_path: Path = Path("models/metrics.json")


__all__ = ["DataConfig", "ModelConfig"]

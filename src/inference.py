"""Inference helpers for serving fraud detection predictions."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import joblib
import pandas as pd

from .pipeline import DEFAULT_DROP_COLUMNS, prepare_features


class FraudModel:
    """Lightweight wrapper around the trained scikit-learn pipeline."""

    def __init__(self, model_path: Path | str = Path("models/fraud_model.joblib")) -> None:
        self.model_path = Path(model_path)
        self.pipeline = joblib.load(self.model_path)

    def predict(self, records: Iterable[Mapping[str, object]]) -> list[int]:
        """Return binary fraud predictions for the provided records."""

        frame = pd.DataFrame(records)
        features = prepare_features(frame, drop_columns=DEFAULT_DROP_COLUMNS)
        return self.pipeline.predict(features).tolist()

    def predict_proba(self, records: Iterable[Mapping[str, object]]) -> list[float]:
        """Return the fraud probability for each record."""

        frame = pd.DataFrame(records)
        features = prepare_features(frame, drop_columns=DEFAULT_DROP_COLUMNS)
        probabilities = self.pipeline.predict_proba(features)[:, 1]
        return probabilities.tolist()


__all__ = ["FraudModel"]

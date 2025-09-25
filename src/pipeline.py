"""Utilities for building the fraud detection machine learning pipeline."""
from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES: tuple[str, ...] = (
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("type",)
DEFAULT_DROP_COLUMNS: tuple[str, ...] = ("nameOrig", "nameDest")


def build_model_pipeline(
    numeric_features: Iterable[str] | None = None,
    categorical_features: Iterable[str] | None = None,
) -> Pipeline:
    """Create the preprocessing + classifier pipeline used during training."""

    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES
    else:
        numeric_features = tuple(numeric_features)
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES
    else:
        categorical_features = tuple(categorical_features)

    transformers = []
    if numeric_features:
        transformers.append(("numeric", StandardScaler(), list(numeric_features)))
    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(categorical_features),
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def prepare_features(
    frame: pd.DataFrame, drop_columns: Iterable[str] | None = None
) -> pd.DataFrame:
    """Return the feature dataframe used by the model."""

    if drop_columns:
        frame = frame.drop(columns=list(drop_columns), errors="ignore")
    return frame.drop(columns=["isFraud"], errors="ignore")


__all__ = [
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "DEFAULT_DROP_COLUMNS",
    "build_model_pipeline",
    "prepare_features",
]

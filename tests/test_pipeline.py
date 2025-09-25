"""Basic tests for the fraud detection pipeline utilities."""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from src.pipeline import build_model_pipeline, prepare_features


def _create_synthetic_dataframe() -> pd.DataFrame:
    data = {
        "step": [1, 2, 3, 4],
        "type": ["PAYMENT", "TRANSFER", "CASH_OUT", "TRANSFER"],
        "amount": [100.0, 250.5, 80.0, 120.0],
        "oldbalanceOrg": [500.0, 1000.0, 600.0, 750.0],
        "newbalanceOrig": [400.0, 749.5, 520.0, 630.0],
        "oldbalanceDest": [0.0, 0.0, 0.0, 50.0],
        "newbalanceDest": [0.0, 250.5, 0.0, 170.0],
        "nameOrig": ["C123", "C456", "C789", "C012"],
        "nameDest": ["M123", "M456", "M789", "M012"],
        "isFlaggedFraud": [0, 0, 0, 1],
        "isFraud": [0, 1, 0, 1],
    }
    return pd.DataFrame(data)


def test_prepare_features_drops_target_and_ids() -> None:
    frame = _create_synthetic_dataframe()
    features = prepare_features(frame, drop_columns=("nameOrig", "nameDest"))
    assert "isFraud" not in features.columns
    assert "nameOrig" not in features.columns
    assert "nameDest" not in features.columns


def test_pipeline_can_fit_on_synthetic_data() -> None:
    frame = _create_synthetic_dataframe()
    features = prepare_features(frame, drop_columns=("nameOrig", "nameDest"))
    pipeline = build_model_pipeline(
        numeric_features=features.select_dtypes(include=[np.number]).columns,
        categorical_features=[col for col in features.columns if features[col].dtype == "object"],
    )
    pipeline.fit(features, frame["isFraud"])
    predictions = pipeline.predict(features)
    assert predictions.shape == (len(frame),)

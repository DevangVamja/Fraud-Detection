"""FastAPI application for serving the fraud detection model."""
from __future__ import annotations

###### if you get an error on src module not found, use the following  lines ######
import sys
import os
TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if TEST_ROOT not in sys.path:
    sys.path.insert(0, TEST_ROOT)

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import FraudModel


class Transaction(BaseModel):
    step: int = Field(..., ge=0)
    type: str
    amount: float = Field(..., ge=0)
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    nameOrig: str | None = None
    nameDest: str | None = None
    isFlaggedFraud: int = Field(0, ge=0)


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float


app = FastAPI(title="Fraud Detection Service", version="1.0.0")
model: FraudModel | None = None


@app.on_event("startup")
def load_model() -> None:
    global model
    model_path = Path("models/fraud_model.joblib")
    if not model_path.exists():
        raise RuntimeError(
            "The trained model could not be found. Run `python -m src.train <data.csv>` first."
        )
    model = FraudModel(model_path=model_path)


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    probabilities = model.predict_proba([transaction.dict()])
    prediction = model.predict([transaction.dict()])
    return PredictionResponse(
        is_fraud=bool(prediction[0]), fraud_probability=float(probabilities[0])
    )


@app.get("/health")
def health() -> dict[str, Any]:
    if model is None:
        status = "model_not_loaded"
    else:
        status = "ok"
    return {"status": status}

"""Training entrypoint for the fraud detection pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from .config import DataConfig, ModelConfig
from .pipeline import DEFAULT_DROP_COLUMNS, build_model_pipeline, prepare_features


def load_data(config: DataConfig) -> pd.DataFrame:
    """Load the dataset from disk using ``pandas``."""

    frame = pd.read_csv(config.csv_path)
    if config.drop_columns:
        frame = frame.drop(columns=config.drop_columns, errors="ignore")
    return frame


def train(config: DataConfig, model_config: ModelConfig) -> dict[str, object]:
    """Train the fraud detection model and persist the artefacts to disk."""

    raw = load_data(config)
    features = prepare_features(
        raw, drop_columns=config.drop_columns or DEFAULT_DROP_COLUMNS
    )
    target = raw[config.target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=model_config.test_size,
        random_state=model_config.random_state,
        stratify=target,
    )

    numeric_columns = features.select_dtypes(include=["number"]).columns
    categorical_columns = [
        column for column in features.columns if features[column].dtype == "object"
    ]

    pipeline = build_model_pipeline(
        numeric_features=numeric_columns, categorical_features=categorical_columns
    )
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]

    metrics: dict[str, object] = classification_report(
        y_test, predictions, output_dict=True
    )
    metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
    metrics["confusion_matrix"] = confusion_matrix(y_test, predictions).tolist()

    model_config.model_output_path.parent.mkdir(parents=True, exist_ok=True)
    model_config.metrics_output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_config.model_output_path)
    with model_config.metrics_output_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fraud detection pipeline")
    parser.add_argument("data", type=Path, help="Path to the CSV dataset")
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/fraud_model.joblib"),
        help="Destination file for the trained pipeline",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("models/metrics.json"),
        help="Where to save the evaluation metrics",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Fraction of data reserved for testing"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed used for train/test split"
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=list(DEFAULT_DROP_COLUMNS),
        help="Columns to drop before training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_config = DataConfig(csv_path=args.data, drop_columns=args.drop_columns)
    model_config = ModelConfig(
        test_size=args.test_size,
        random_state=args.random_state,
        model_output_path=args.model_output,
        metrics_output_path=args.metrics_output,
    )
    metrics = train(data_config, model_config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

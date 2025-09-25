# Fraud Detection Using Scikit-Learn

## Overview
This project provides an end-to-end workflow for detecting fraudulent financial transactions. It covers feature preparation, model training with a scikit-learn pipeline, automated tests, and a FastAPI inference service for real-time scoring.

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training Pipeline](#training-pipeline)
- [Running Tests](#running-tests)
- [Serving Predictions](#serving-predictions)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features
- Reproducible feature engineering and model pipeline built around `ColumnTransformer` and `LogisticRegression`.
- CLI training script (`src/train.py`) that exports the trained pipeline and evaluation metrics.
- Unit tests for the data-preparation and modelling utilities.
- FastAPI application (`app/main.py`) for serving predictions and a simple web UI.
- Dockerfile for containerised deployments.

## Dataset
The repository no longer ships with raw data. Download the PaySim dataset from Kaggle:

- https://www.kaggle.com/datasets/ealaxi/paysim1

Place the CSV (for example `PS_20174392719_1491204439457_log.csv`) inside the `data/` directory before running the training script.

Key columns used during training:
- `step`, `type`, `amount`
- Balances before/after the transaction (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`)
- `isFlaggedFraud` (system flag)
- `isFraud` (target label)

Transaction IDs (`nameOrig`, `nameDest`) are dropped by default.

## Installation
```bash
git clone https://github.com/DevangVamja/Fraud-Detection.git
cd Fraud-Detection
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## Training Pipeline
Run the training entrypoint with the dataset path. The script splits the data, fits the preprocessing + logistic regression pipeline, and saves artefacts under `models/`.

```bash
python -m src.train data/PS_20174392719_1491204439457_log.csv \
  --model-output models/fraud_model.joblib \
  --metrics-output models/metrics.json
```

Optional flags:
- `--test-size`: hold-out fraction (default `0.2`).
- `--random-state`: seed for reproducibility (default `42`).
- `--drop-columns`: override the columns removed before training (defaults to `nameOrig nameDest`).

After execution you will find the fitted pipeline (`fraud_model.joblib`) and evaluation metrics (`metrics.json`).

## Running Tests
Unit tests validate the feature preparation and modelling helpers on synthetic data:

```bash
pytest
```

## Serving Predictions
Use the FastAPI service once a trained model exists at `models/fraud_model.joblib`:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Example Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "step": 1,
        "type": "TRANSFER",
        "amount": 1500.0,
        "oldbalanceOrg": 2000.0,
        "newbalanceOrig": 500.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 1500.0,
        "isFlaggedFraud": 0
      }'
```

Health checks are available at `GET /health`.

## Configuration
Runtime configuration is defined in `src/config.py` (data locations, default drop columns, and model settings). Adjust these defaults or pass CLI flags when needed.

## Project Structure
```
.
|-- app/                # FastAPI service and templates
|-- models/             # Saved pipelines and metrics (generated)
|-- src/                # Training, inference, and pipeline utilities
|-- tests/              # Unit tests
|-- Notebooks/          # Exploratory notebooks
|-- Dockerfile
|-- requirements.txt
`-- README.md
```

## Contributing
1. Fork the repository and create a new branch for your feature or bug fix.
2. Run the tests (`pytest`) before opening a pull request.
3. Describe the changes clearly in the PR body.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Contact
- **Devang Vamja** - https://devangvamja-portfolio.netlify.app/
- Email - devangvamja2000@gmail.com
- LinkedIn - https://linkedin.com/in/DevangVamja
- GitHub - https://github.com/DevangVamja

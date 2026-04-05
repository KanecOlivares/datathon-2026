"""Evaluation helpers."""

from __future__ import annotations

import argparse
import importlib
from typing import Any

import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.config import (
    ACADEMIC_FACTORS_PROCESSED_PATH,
    ACADEMIC_FACTORS_TARGET_COLUMN,
    METRICS_PATH,
    MODEL_PATH,
)


def evaluate_mae(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return mean absolute error for regression predictions."""
    return {"mae": float(mean_absolute_error(y_true, y_pred))}


def evaluate_summary(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return the standard evaluation payload for the baseline regressor."""
    return evaluate_mae(y_true, y_pred)


def evaluate_saved_model(
    data_path: str = str(ACADEMIC_FACTORS_PROCESSED_PATH),
    model_path: str = str(MODEL_PATH),
) -> dict[str, float]:
    """Load the saved model and evaluate it on the deterministic test split."""
    train_module = importlib.import_module("src.08_train")
    df = pd.read_csv(data_path)
    training_artifacts = train_module.train_model(df, target_column=ACADEMIC_FACTORS_TARGET_COLUMN)
    saved_model = train_module.load_model(model_path)
    return train_module.test_model(
        saved_model,
        training_artifacts["X_test"],
        training_artifacts["y_test"],
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evaluation entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate the saved random-forest regressor.")
    parser.add_argument(
        "--data-path",
        default=str(ACADEMIC_FACTORS_PROCESSED_PATH),
        help="Path to the processed evaluation dataset.",
    )
    parser.add_argument(
        "--model-path",
        default=str(MODEL_PATH),
        help="Path to the saved model artifact.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for evaluating the saved baseline model."""
    args = parse_args()
    metrics = evaluate_saved_model(data_path=args.data_path, model_path=args.model_path)
    print(f"Evaluated saved model from {MODEL_PATH} with MAE={metrics['mae']:.4f}.")


if __name__ == "__main__":
    main()

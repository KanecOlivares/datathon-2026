"""Training entrypoints for the baseline random-forest regressor."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    ACADEMIC_FACTORS_CATEGORICAL_COLUMNS,
    ACADEMIC_FACTORS_LEAKAGE_COLUMNS,
    ACADEMIC_FACTORS_NUMERIC_COLUMNS,
    ACADEMIC_FACTORS_PROCESSED_PATH,
    ACADEMIC_FACTORS_TARGET_COLUMN,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_SEED,
)
from src.utils import ensure_directory, write_json
from src.validation import split_data, validate_training_columns


def load_training_dataframe(path: str = str(ACADEMIC_FACTORS_PROCESSED_PATH)) -> pd.DataFrame:
    """Load the processed dataset used for model training."""
    return pd.read_csv(path)


def get_training_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Select non-leaking raw and engineered features for model training."""
    categorical_columns = [column for column in ACADEMIC_FACTORS_CATEGORICAL_COLUMNS if column in df.columns]
    numeric_candidates = set(ACADEMIC_FACTORS_NUMERIC_COLUMNS) | {
        "Study_Attendance_Product",
        "Study_Sleep_Balance",
        "Academic_Support_Index",
        "Home_Support_Index",
        "Wellbeing_Index",
    }
    numeric_columns = sorted(
        column
        for column in numeric_candidates
        if column in df.columns and column != ACADEMIC_FACTORS_TARGET_COLUMN
    )
    excluded_columns = set(ACADEMIC_FACTORS_LEAKAGE_COLUMNS)
    selected_columns = [
        column for column in categorical_columns + numeric_columns if column not in excluded_columns
    ]
    return selected_columns, categorical_columns


def build_preprocessing_pipeline(
    categorical_columns: list[str], numeric_columns: list[str]
) -> ColumnTransformer:
    """Create the shared preprocessing transformer for training and inference."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
            ("numeric", "passthrough", numeric_columns),
        ]
    )


def build_model_pipeline(
    categorical_columns: list[str], numeric_columns: list[str]
) -> Pipeline:
    """Create the full preprocessing-plus-model pipeline."""
    preprocessing = build_preprocessing_pipeline(categorical_columns, numeric_columns)
    regressor = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessing),
            ("model", regressor),
        ]
    )


def train_model(df: pd.DataFrame, target_column: str = ACADEMIC_FACTORS_TARGET_COLUMN) -> dict[str, Any]:
    """Train the baseline regressor and return the fitted artifacts."""
    feature_columns, categorical_columns = get_training_feature_columns(df)
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    validate_training_columns(X)

    X_train, X_test, y_train, y_test = split_data(
        pd.concat([X, y], axis=1),
        target_column=target_column,
    )

    numeric_columns = [column for column in X_train.columns if column not in categorical_columns]
    pipeline = build_model_pipeline(categorical_columns, numeric_columns)
    pipeline.fit(X_train, y_train)

    return {
        "model": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
    }


def predict_model(model: Pipeline, X: pd.DataFrame) -> pd.Series:
    """Generate predictions from the trained pipeline."""
    predictions = model.predict(X)
    return pd.Series(predictions, index=X.index, name="prediction")


def test_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evaluate the trained pipeline on the held-out test split."""
    evaluate_module = importlib.import_module("src.09_evaluate")
    y_pred = predict_model(model, X_test)
    return evaluate_module.evaluate_summary(y_test, y_pred)


def evaluate_train_test_split(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Return comparable MAE metrics for both training and test splits."""
    evaluate_module = importlib.import_module("src.09_evaluate")
    train_predictions = predict_model(model, X_train)
    test_predictions = predict_model(model, X_test)
    return {
        "train_mae": evaluate_module.evaluate_mae(y_train, train_predictions)["mae"],
        "test_mae": evaluate_module.evaluate_mae(y_test, test_predictions)["mae"],
    }


def save_model(model: Pipeline, path: str = str(MODEL_PATH)) -> None:
    """Persist the trained pipeline."""
    destination = Path(path)
    ensure_directory(destination.parent)
    joblib.dump(model, destination)


def load_model(path: str = str(MODEL_PATH)) -> Pipeline:
    """Load a persisted trained pipeline."""
    return joblib.load(path)


def train_and_save_model(df: pd.DataFrame, target_column: str = ACADEMIC_FACTORS_TARGET_COLUMN) -> dict[str, Any]:
    """Train the model, evaluate it, and persist the artifacts."""
    training_artifacts = train_model(df, target_column=target_column)
    metrics = evaluate_train_test_split(
        training_artifacts["model"],
        training_artifacts["X_train"],
        training_artifacts["y_train"],
        training_artifacts["X_test"],
        training_artifacts["y_test"],
    )
    save_model(training_artifacts["model"])
    write_json(
        {
            "model_type": "RandomForestRegressor",
            "random_seed": RANDOM_SEED,
            "target_column": target_column,
            "feature_columns": training_artifacts["feature_columns"],
            "categorical_columns": training_artifacts["categorical_columns"],
            "numeric_columns": training_artifacts["numeric_columns"],
            "train_test_metrics": metrics,
        },
        METRICS_PATH,
    )
    return {**training_artifacts, "metrics": metrics}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training entrypoint."""
    parser = argparse.ArgumentParser(description="Train the baseline random-forest regressor.")
    parser.add_argument(
        "--data-path",
        default=str(ACADEMIC_FACTORS_PROCESSED_PATH),
        help="Path to the processed training dataset.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for baseline model training."""
    args = parse_args()
    df = load_training_dataframe(args.data_path)
    training_artifacts = train_and_save_model(df)
    print(
        f"Saved RandomForestRegressor to {MODEL_PATH} with "
        f"train_MAE={training_artifacts['metrics']['train_mae']:.4f} and "
        f"test_MAE={training_artifacts['metrics']['test_mae']:.4f}."
    )


if __name__ == "__main__":
    main()

"""Validation helpers for train and evaluation splits."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ACADEMIC_FACTORS_CATEGORICAL_COLUMNS,
    ACADEMIC_FACTORS_NUMERIC_COLUMNS,
    ACADEMIC_FACTORS_REQUIRED_COLUMNS,
)
from src.config import RANDOM_SEED


ACADEMIC_FACTORS_ALLOWED_VALUES = {
    "Parental_Involvement": {"Low", "Medium", "High"},
    "Access_to_Resources": {"Low", "Medium", "High"},
    "Extracurricular_Activities": {"No", "Yes"},
    "Motivation_Level": {"Low", "Medium", "High"},
    "Internet_Access": {"No", "Yes"},
    "Family_Income": {"Low", "Medium", "High"},
    "Teacher_Quality": {"Low", "Medium", "High"},
    "School_Type": {"Public", "Private"},
    "Peer_Influence": {"Negative", "Neutral", "Positive"},
    "Learning_Disabilities": {"No", "Yes"},
    "Parental_Education_Level": {"High School", "College", "Postgraduate"},
    "Distance_from_Home": {"Near", "Moderate", "Far"},
    "Gender": {"Female", "Male"},
}


def validate_required_columns(df: pd.DataFrame) -> None:
    """Ensure the dataframe contains the expected dataset schema."""
    missing_columns = sorted(set(ACADEMIC_FACTORS_REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def validate_numeric_columns(df: pd.DataFrame) -> None:
    """Ensure numeric fields can be interpreted as numbers."""
    invalid_columns = []
    for column in ACADEMIC_FACTORS_NUMERIC_COLUMNS:
        try:
            pd.to_numeric(df[column], errors="raise")
        except Exception:
            invalid_columns.append(column)
    if invalid_columns:
        raise ValueError(f"Non-numeric values found in numeric columns: {invalid_columns}")


def validate_categorical_values(df: pd.DataFrame) -> None:
    """Ensure categorical fields only contain known labels, excluding nulls."""
    invalid_values: dict[str, list[str]] = {}
    for column in ACADEMIC_FACTORS_CATEGORICAL_COLUMNS:
        observed = set(df[column].dropna().astype(str).str.strip().unique())
        unexpected = sorted(observed - ACADEMIC_FACTORS_ALLOWED_VALUES[column])
        if unexpected:
            invalid_values[column] = unexpected
    if invalid_values:
        raise ValueError(f"Unexpected categorical values detected: {invalid_values}")


def validate_no_missing_values(df: pd.DataFrame) -> None:
    """Ensure downstream-ready data does not contain nulls."""
    missing_counts = df.isna().sum()
    remaining = missing_counts[missing_counts > 0]
    if not remaining.empty:
        raise ValueError(f"Missing values remain after preprocessing: {remaining.to_dict()}")


def validate_academic_factors_dataframe(df: pd.DataFrame) -> None:
    """Run dataset-specific validation checks."""
    validate_required_columns(df)
    validate_numeric_columns(df)
    validate_categorical_values(df)


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target using a fixed random seed."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y)

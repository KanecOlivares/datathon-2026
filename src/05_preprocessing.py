"""Preprocessing entry points."""

from __future__ import annotations

import pandas as pd

from src.config import ACADEMIC_FACTORS_CATEGORICAL_COLUMNS, ACADEMIC_FACTORS_NUMERIC_COLUMNS


def _clean_string_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Normalize whitespace in categorical columns without changing labels."""
    cleaned = df.copy()
    for column in columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()
    return cleaned


def _impute_missing_categoricals(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fill categorical nulls with the mode of each column."""
    imputed = df.copy()
    for column in columns:
        if imputed[column].isna().any():
            mode = imputed[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            imputed[column] = imputed[column].fillna(fill_value)
    return imputed


def _coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Force numeric columns to numeric dtypes and fail loudly on bad values."""
    coerced = df.copy()
    for column in columns:
        coerced[column] = pd.to_numeric(coerced[column], errors="raise")
    return coerced


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw academic factors data into a stable tabular form."""
    processed = _clean_string_columns(df, ACADEMIC_FACTORS_CATEGORICAL_COLUMNS)
    processed = _impute_missing_categoricals(processed, ACADEMIC_FACTORS_CATEGORICAL_COLUMNS)
    processed = _coerce_numeric_columns(processed, ACADEMIC_FACTORS_NUMERIC_COLUMNS)
    return processed

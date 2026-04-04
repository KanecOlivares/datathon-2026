"""Validation helpers for train and evaluation splits."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target using a fixed random seed."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y)

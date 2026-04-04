"""Preprocessing entry points."""

from __future__ import annotations

import pandas as pd


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy so preprocessing steps can be added safely."""
    return df.copy()

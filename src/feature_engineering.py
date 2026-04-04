"""Feature engineering utilities."""

from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy so feature logic can be layered in incrementally."""
    return df.copy()

"""Data loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ACADEMIC_FACTORS_RAW_PATH
from src.utils import write_dataframe


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load a CSV file into a dataframe."""
    return pd.read_csv(path, **kwargs)


def load_academic_factors(path: str | Path = ACADEMIC_FACTORS_RAW_PATH) -> pd.DataFrame:
    """Load the academic factors dataset from the default raw-data location."""
    return load_csv(path)


def save_dataset(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """Persist a dataframe to a CSV file."""
    write_dataframe(df, Path(path), **kwargs)

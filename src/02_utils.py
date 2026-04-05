"""Shared utility helpers for the datathon pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write a JSON file with stable formatting."""
    ensure_directory(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_dataframe(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    """Persist a dataframe to CSV, creating parent directories first."""
    ensure_directory(path.parent)
    df.to_csv(path, index=False, **kwargs)

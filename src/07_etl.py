"""End-to-end ETL runner for the academic factors dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import (
    ACADEMIC_FACTORS_PROCESSED_PATH,
    ACADEMIC_FACTORS_RAW_PATH,
    ACADEMIC_FACTORS_SUMMARY_PATH,
)
from src.feature_engineering import build_features
from src.ingestion import load_academic_factors, save_dataset
from src.preprocessing import preprocess_dataframe
from src.utils import write_json
from src.validation import validate_academic_factors_dataframe, validate_no_missing_values


def build_etl_summary(raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> dict[str, Any]:
    """Capture a compact summary of the ETL run."""
    raw_missing = raw_df.isna().sum()
    processed_missing = processed_df.isna().sum()
    return {
        "source_path": str(ACADEMIC_FACTORS_RAW_PATH),
        "output_path": str(ACADEMIC_FACTORS_PROCESSED_PATH),
        "raw_rows": int(raw_df.shape[0]),
        "raw_columns": int(raw_df.shape[1]),
        "processed_rows": int(processed_df.shape[0]),
        "processed_columns": int(processed_df.shape[1]),
        "raw_missing_values": {k: int(v) for k, v in raw_missing[raw_missing > 0].to_dict().items()},
        "processed_missing_values": {
            k: int(v) for k, v in processed_missing[processed_missing > 0].to_dict().items()
        },
        "derived_columns": sorted(set(processed_df.columns) - set(raw_df.columns)),
    }


def run_academic_factors_etl() -> pd.DataFrame:
    """Load, validate, clean, feature engineer, and save the dataset."""
    raw_df = load_academic_factors()
    validate_academic_factors_dataframe(raw_df)

    cleaned_df = preprocess_dataframe(raw_df)
    featured_df = build_features(cleaned_df)
    validate_no_missing_values(featured_df)

    save_dataset(featured_df, ACADEMIC_FACTORS_PROCESSED_PATH)
    write_json(build_etl_summary(raw_df, featured_df), ACADEMIC_FACTORS_SUMMARY_PATH)
    return featured_df


def main() -> None:
    """Run ETL as a module entrypoint."""
    df = run_academic_factors_etl()
    print(
        f"Saved processed dataset to {ACADEMIC_FACTORS_PROCESSED_PATH} "
        f"with shape {df.shape[0]} rows x {df.shape[1]} columns."
    )


if __name__ == "__main__":
    main()

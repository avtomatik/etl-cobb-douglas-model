import pandas as pd


def validate_columns(df: pd.DataFrame, required_columns: set) -> None:
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Plot contract violated. Missing columns: {missing}")

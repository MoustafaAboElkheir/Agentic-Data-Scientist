"""
EDA Tools for the Agentic Data Scientist pipeline.
Provides dataset profiling, missing value analysis, and distribution inspection.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Generate a comprehensive profile of a pandas DataFrame.
    Returns a dictionary with shape, dtypes, missing values, and basic statistics.
    """
    logger.info(f"Profiling dataset with shape {df.shape}")
    profile = {
        "shape": df.shape,
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "numeric_stats": df.describe().round(4).to_dict(),
        "target_distribution": None,
    }
    return profile


def detect_target_column(df: pd.DataFrame) -> str:
    """
    Heuristically detect the target column in a DataFrame.
    Prefers columns named 'target', 'label', 'class', or the last column.
    """
    candidates = ['target', 'label', 'class', 'y', 'output']
    for col in candidates:
        if col in df.columns:
            logger.info(f"Detected target column: '{col}'")
            return col
    logger.info(f"No standard target column found; using last column: '{df.columns[-1]}'")
    return df.columns[-1]


def summarize_class_balance(y: pd.Series) -> dict:
    """Return class counts and percentages for a target series."""
    counts = y.value_counts().to_dict()
    pcts = (y.value_counts(normalize=True) * 100).round(2).to_dict()
    return {"counts": counts, "percentages": pcts}

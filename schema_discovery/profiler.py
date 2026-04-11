"""
Statistical profiler: computes per-column metrics used as input for the AI classifier.
"""
import re
import numpy as np
import pandas as pd
from typing import Any


_DATE_RE = re.compile(
    r"\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}|\d{4}-\d{2}-\d{2}T\d{2}"
)


def profile_dataframe(df: pd.DataFrame) -> dict:
    """
    Return a nested dict with:
      - shape: {rows, columns}
      - columns: {col_name: column_profile_dict, ...}
    """
    return {
        "shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
        "columns": {col: _profile_column(df[col]) for col in df.columns},
    }


def _profile_column(series: pd.Series) -> dict:
    n = len(series)
    null_count = int(series.isna().sum())
    unique_count = int(series.nunique(dropna=True))

    profile: dict[str, Any] = {
        "dtype": str(series.dtype),
        "null_count": null_count,
        "null_rate": round(null_count / max(n, 1), 4),
        "unique_count": unique_count,
        "unique_rate": round(unique_count / max(n, 1), 4),
        "sample_values": _sample_values(series),
    }

    # Numeric stats
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        s = series.dropna()
        if len(s) > 0:
            profile["min"] = _scalar(s.min())
            profile["max"] = _scalar(s.max())
            profile["mean"] = round(float(s.mean()), 6)
            profile["std"] = round(float(s.std()), 6)
            profile["median"] = round(float(s.median()), 6)
            profile["q25"] = round(float(s.quantile(0.25)), 6)
            profile["q75"] = round(float(s.quantile(0.75)), 6)
            profile["negative_count"] = int((s < 0).sum())
            profile["zero_count"] = int((s == 0).sum())
            # Check if all non-null values are whole numbers
            try:
                profile["all_integers"] = bool((s == s.round(0)).all())
            except Exception:
                profile["all_integers"] = False

    # String stats
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        s = series.dropna().astype(str)
        if len(s) > 0:
            lengths = s.str.len()
            profile["avg_length"] = round(float(lengths.mean()), 2)
            profile["max_length"] = int(lengths.max())
            profile["looks_like_dates"] = _looks_like_dates(
                s.head(20).tolist()
            )

    # Datetime stats
    if pd.api.types.is_datetime64_any_dtype(series):
        s = series.dropna()
        if len(s) > 0:
            profile["min_date"] = str(s.min())
            profile["max_date"] = str(s.max())

    # Boolean stats
    if pd.api.types.is_bool_dtype(series):
        s = series.dropna()
        if len(s) > 0:
            profile["true_rate"] = round(float(s.mean()), 4)

    return profile


def _sample_values(series: pd.Series, n: int = 8) -> list:
    unique = series.dropna().unique()
    sample = unique[:n]
    return [_scalar(v) for v in sample]


def _scalar(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if np.isnan(f) or np.isinf(f) else round(f, 6)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    return v


def _looks_like_dates(samples: list[str]) -> bool:
    if not samples:
        return False
    matches = sum(1 for s in samples if _DATE_RE.search(s))
    return matches >= max(1, len(samples) * 0.6)

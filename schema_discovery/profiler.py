"""
Two-pass profiler.

Pass 1 — lightweight_profile_dataframe()
    Minimal stats only (dtype, nulls, cardinality, sample values).
    Used as input for Gemini's first-pass category detection.
    Fast and cheap — no heavy computation.

Pass 2 — targeted_profile()
    Called per-column AFTER Gemini has identified the category.
    Computes only the stats that are meaningful for that category.
    Identifiers get nothing. Temporals get date ranges. Categoricals
    get value counts. Etc.
"""
import re
import math
import numpy as np
import pandas as pd
from typing import Any


_DATE_RE = re.compile(
    r"\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}|\d{4}-\d{2}-\d{2}T\d{2}"
)

# Maps category → which stat groups to compute in Pass 2
_STAT_GROUPS: dict[str, list[str]] = {
    "numeric_continuous":  ["distribution", "range", "negative"],
    "numeric_discrete":    ["distribution", "range", "negative", "zero", "value_counts"],
    "numeric_ratio":       ["distribution", "range", "out_of_range"],
    "numeric_amount":      ["distribution", "range", "negative", "zero"],
    "temporal":            ["date_range"],
    "categorical_nominal": ["value_counts", "entropy"],
    "categorical_ordinal": ["value_counts", "entropy"],
    "boolean":             ["boolean_rates"],
    "textual":             ["text_length", "word_count"],
    "identifier":          [],   # IDs have no meaningful stats
    "geographic":          ["text_length", "value_counts"],
    "unknown":             ["distribution", "range", "text_length"],
}


# ──────────────────────────────────────────────
# Pass 1
# ──────────────────────────────────────────────

def lightweight_profile_dataframe(df: pd.DataFrame) -> dict:
    """Minimal profile for Gemini's first-pass category detection."""
    return {
        "shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
        "columns": {col: _lightweight_profile(df[col]) for col in df.columns},
    }


def _lightweight_profile(series: pd.Series) -> dict:
    n = len(series)
    null_count = int(series.isna().sum())
    unique_count = int(series.nunique(dropna=True))
    return {
        "dtype": str(series.dtype),
        "null_rate": round(null_count / max(n, 1), 4),
        "unique_count": unique_count,
        "unique_rate": round(unique_count / max(n, 1), 4),
        "sample_values": _sample_values(series),
    }


# ──────────────────────────────────────────────
# Pass 2
# ──────────────────────────────────────────────

def targeted_profile(series: pd.Series, category: str) -> dict:
    """
    Compute only the stats relevant to the detected category.
    Returns an empty dict for identifiers.
    """
    groups = _STAT_GROUPS.get(category.lower(), [])
    stats: dict[str, Any] = {}
    for group in groups:
        stats.update(_compute_group(series, group))
    return stats


def _compute_group(series: pd.Series, group: str) -> dict:
    result: dict[str, Any] = {}
    is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    s_num = series.dropna() if is_numeric else None

    if group == "distribution":
        if s_num is not None and len(s_num) > 0:
            result["mean"]   = round(float(s_num.mean()), 6)
            result["std"]    = round(float(s_num.std()), 6)
            result["median"] = round(float(s_num.median()), 6)
            result["q25"]    = round(float(s_num.quantile(0.25)), 6)
            result["q75"]    = round(float(s_num.quantile(0.75)), 6)

    elif group == "range":
        if s_num is not None and len(s_num) > 0:
            result["min"] = _scalar(s_num.min())
            result["max"] = _scalar(s_num.max())

    elif group == "negative":
        if s_num is not None:
            result["negative_count"] = int((s_num < 0).sum())

    elif group == "zero":
        if s_num is not None:
            result["zero_count"] = int((s_num == 0).sum())

    elif group == "out_of_range":
        if s_num is not None and len(s_num) > 0:
            result["pct_below_0"] = round(float((s_num < 0).mean()), 4)
            result["pct_above_1"] = round(float((s_num > 1).mean()), 4)

    elif group == "value_counts":
        counts = series.value_counts(dropna=True).head(20)
        result["top_values"] = {str(k): int(v) for k, v in counts.items()}

    elif group == "entropy":
        probs = series.value_counts(dropna=True, normalize=True)
        if len(probs) > 0:
            result["entropy"] = round(
                -sum(p * math.log2(p) for p in probs if p > 0), 4
            )

    elif group == "boolean_rates":
        s = series.dropna()
        if len(s) > 0:
            try:
                vals = s.astype(bool)
                result["true_rate"]  = round(float(vals.mean()), 4)
                result["false_rate"] = round(1.0 - float(vals.mean()), 4)
            except Exception:
                pass

    elif group == "date_range":
        s = series.dropna()
        if pd.api.types.is_object_dtype(series):
            try:
                s = pd.to_datetime(s, infer_datetime_format=True, errors="coerce").dropna()
            except Exception:
                pass
        if pd.api.types.is_datetime64_any_dtype(s) and len(s) > 0:
            result["min_date"]        = str(s.min())
            result["max_date"]        = str(s.max())
            result["date_range_days"] = int((s.max() - s.min()).days)

    elif group == "text_length":
        s = series.dropna().astype(str)
        if len(s) > 0:
            lengths = s.str.len()
            result["avg_length"] = round(float(lengths.mean()), 2)
            result["max_length"] = int(lengths.max())
            result["min_length"] = int(lengths.min())

    elif group == "word_count":
        s = series.dropna().astype(str)
        if len(s) > 0:
            wc = s.str.split().str.len()
            result["avg_word_count"] = round(float(wc.mean()), 2)
            result["max_word_count"] = int(wc.max())

    return result


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _sample_values(series: pd.Series, n: int = 8) -> list:
    unique = series.dropna().unique()
    return [_scalar(v) for v in unique[:n]]


def _scalar(v: Any) -> Any:
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        f = float(v)
        return None if np.isnan(f) or np.isinf(f) else round(f, 6)
    if isinstance(v, np.bool_):
        return bool(v)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v

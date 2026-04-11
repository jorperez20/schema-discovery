"""
Rule-based pre-classifier.
Generates domain hints from column name patterns and statistical profile.
These hints are sent to the AI model as soft signals, not hard decisions.
"""
from __future__ import annotations

_NAME_HINTS: dict[str, list[str]] = {
    "boolean":     ["is_", "has_", "_flag", "_bool", "active", "enabled",
                    "deleted", "verified", "approved", "valid", "exists"],
    "identifier":  ["_id", "id_", "_uuid", "_key", "_code", "_ref", "_sku",
                    "barcode", "serial", "token", "hash", "guid", "pk"],
    "temporal":    ["_date", "_time", "_at", "_on", "_ts", "timestamp",
                    "created", "updated", "modified", "deleted", "started",
                    "ended", "year", "month", "day", "hour", "minute"],
    "geographic":  ["lat", "lon", "lng", "latitude", "longitude", "zip",
                    "postal", "country", "city", "state", "region", "address",
                    "street", "province", "territory"],
    "amount":      ["amount", "price", "cost", "revenue", "sales", "fee",
                    "wage", "salary", "payment", "balance", "total", "sum",
                    "budget", "spend", "earning", "income", "charge", "bill"],
    "ratio":       ["rate", "ratio", "pct", "percent", "percentage", "share",
                    "proportion", "score", "weight", "prob", "probability",
                    "likelihood", "confidence", "accuracy"],
    "categorical": ["category", "type", "status", "gender", "class", "group",
                    "segment", "tier", "level", "rank", "grade", "label",
                    "tag", "brand", "channel", "source", "medium", "format"],
    "free_text":   ["comment", "description", "note", "text", "message",
                    "content", "body", "review", "feedback", "remark",
                    "summary", "detail", "narrative"],
}


def pre_classify(col_name: str, profile: dict) -> str | None:
    """
    Return a domain hint string (e.g. 'amount (likely)') or None.
    Hints are best-effort — the AI model makes the final decision.
    """
    name = col_name.lower().replace(" ", "_")

    # --- dtype-based (high confidence) ---
    dtype = profile.get("dtype", "")
    if dtype.startswith("bool"):
        return "boolean"
    if dtype.startswith("datetime"):
        return "temporal"

    # --- looks like dates in object column ---
    if dtype == "object" and profile.get("looks_like_dates"):
        return "temporal (likely)"

    # --- name-based ---
    for domain, keywords in _NAME_HINTS.items():
        for kw in keywords:
            if kw in name:
                return f"{domain} (likely)"

    # --- cardinality-based ---
    unique_count = profile.get("unique_count", 0)
    unique_rate = profile.get("unique_rate", 0.0)
    null_rate = profile.get("null_rate", 0.0)

    if unique_count <= 2 and null_rate < 0.5:
        return "boolean (likely)"

    if unique_rate < 0.05 and unique_count <= 50:
        return "categorical (likely)"

    if unique_rate > 0.95 and unique_count > 100:
        return "identifier (likely)"

    # --- numeric range-based ---
    mn = profile.get("min")
    mx = profile.get("max")
    all_ints = profile.get("all_integers", True)
    if mn is not None and mx is not None:
        try:
            if 0.0 <= float(mn) and float(mx) <= 1.0 and not all_ints:
                return "ratio (likely)"
            if float(mx) <= 100.0 and float(mn) >= 0.0 and not all_ints:
                return "ratio or continuous (likely)"
        except (TypeError, ValueError):
            pass

    return None

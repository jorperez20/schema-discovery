"""
SchemaDiscoveryResult: holds and displays schema discovery output.

In Jupyter notebooks, the result renders as a styled HTML table automatically.
Programmatic access is available via .to_dataframe() and dict-style access.
"""
from __future__ import annotations

import pandas as pd

_CATEGORY_PALETTE: dict[str, tuple[str, str]] = {
    # category                bg_color    text_color
    "numeric_continuous":   ("#4E79A7", "#fff"),
    "numeric_discrete":     ("#6A9DC8", "#fff"),
    "numeric_ratio":        ("#B07AA1", "#fff"),
    "numeric_amount":       ("#F28E2B", "#fff"),
    "temporal":             ("#9C755F", "#fff"),
    "categorical_nominal":  ("#59A14F", "#fff"),
    "categorical_ordinal":  ("#8BC34A", "#fff"),
    "boolean":              ("#26A69A", "#fff"),
    "textual":              ("#EDC948", "#333"),
    "identifier":           ("#BAB0AC", "#333"),
    "geographic":           ("#86BCB6", "#333"),
    "unknown":              ("#cccccc", "#333"),
}


class SchemaDiscoveryResult:
    """
    Container for schema discovery output.

    Attributes
    ----------
    source_name : str
        Name of the data source (file path or user-supplied label).
    shape : dict
        {'rows': int, 'columns': int}
    profile : dict
        Raw statistical profile per column.
    classifications : dict
        AI classifications per column keyed by column name.
    """

    def __init__(
        self,
        source_name: str,
        shape: dict,
        profile: dict,
        classifications: list[dict],
    ):
        self.source_name = source_name
        self.shape = shape
        self.profile = profile
        self.classifications = {c["column"]: c for c in classifications}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return a flat DataFrame with one row per column."""
        rows = []
        for col, p in self.profile.items():
            clf = self.classifications.get(col, {})
            rows.append(
                {
                    "column":        col,
                    "dtype":         p.get("dtype"),
                    "category":      clf.get("category", "unknown"),
                    "domain":        clf.get("domain", "unknown"),
                    "semantic_type": clf.get("semantic_type"),
                    "null_rate":     p.get("null_rate"),
                    "unique_count":  p.get("unique_count"),
                    "unique_rate":   p.get("unique_rate"),
                    "sample_values": p.get("sample_values", []),
                    "notes":         clf.get("notes"),
                }
            )
        return pd.DataFrame(rows)

    def columns_by_category(self, category: str) -> list[str]:
        """Return column names whose broad category matches (case-insensitive).

        Categories: numeric | temporal | categorical | textual | identifier | geographic
        """
        target = category.lower()
        return [
            col
            for col, clf in self.classifications.items()
            if clf.get("category", "").lower() == target
        ]

    def columns_by_domain(self, domain: str) -> list[str]:
        """Return column names whose inferred domain contains `domain` (case-insensitive substring)."""
        target = domain.lower()
        return [
            col
            for col, clf in self.classifications.items()
            if target in clf.get("domain", "").lower()
        ]

    @property
    def category_summary(self) -> pd.Series:
        """Value counts of broad categories across all columns."""
        cats = [c.get("category", "unknown") for c in self.classifications.values()]
        return pd.Series(cats, name="category").value_counts()

    @property
    def domain_summary(self) -> pd.DataFrame:
        """All inferred domains with their category, sorted by category."""
        rows = [
            {"column": col, "category": clf.get("category", "unknown"), "domain": clf.get("domain", "unknown")}
            for col, clf in self.classifications.items()
        ]
        return pd.DataFrame(rows).sort_values("category").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Jupyter display
    # ------------------------------------------------------------------

    def _repr_html_(self) -> str:
        df = self.to_dataframe()
        rows_html = []

        for _, row in df.iterrows():
            category = row["category"] or "unknown"
            bg, fg = _CATEGORY_PALETTE.get(category, _CATEGORY_PALETTE["unknown"])

            null_pct = (
                f"{row['null_rate'] * 100:.1f}%"
                if pd.notna(row["null_rate"])
                else "—"
            )
            uniq_str = "—"
            if pd.notna(row["unique_count"]) and pd.notna(row["unique_rate"]):
                uniq_str = (
                    f"{int(row['unique_count']):,}"
                    f" <span style='color:#aaa'>({row['unique_rate']*100:.1f}%)</span>"
                )

            samples_list = row["sample_values"] or []
            samples_str = ", ".join(str(v) for v in samples_list[:5])
            samples_title = ", ".join(str(v) for v in samples_list)

            domain   = row["domain"] or ""
            semantic = row["semantic_type"] or ""
            notes    = row["notes"] or ""

            category_badge = (
                f'<span style="background:{bg};color:{fg};'
                f'padding:2px 10px;border-radius:12px;'
                f'font-size:0.8em;white-space:nowrap;font-weight:600">'
                f"{category}</span>"
            )

            rows_html.append(
                f"<tr>"
                f'<td style="font-weight:600;white-space:nowrap">{row["column"]}</td>'
                f'<td style="color:#888;font-size:0.82em;white-space:nowrap">{row["dtype"]}</td>'
                f'<td>{category_badge}</td>'
                f'<td style="font-size:0.82em;color:#555;font-family:monospace">{domain}</td>'
                f'<td style="font-size:0.82em">{semantic}</td>'
                f'<td style="text-align:center;font-size:0.82em">{null_pct}</td>'
                f'<td style="text-align:right;font-size:0.82em">{uniq_str}</td>'
                f'<td style="font-size:0.78em;color:#666;max-width:220px;'
                f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap" '
                f'title="{samples_title}">{samples_str}</td>'
                f'<td style="font-size:0.78em;color:#888;max-width:200px">{notes}</td>'
                f"</tr>"
            )

        # Legend
        legend_items = "".join(
            f'<span style="background:{bg};color:{fg};padding:2px 8px;'
            f'border-radius:10px;font-size:0.75em;margin:2px 3px;display:inline-block">'
            f"{cat}</span>"
            for cat, (bg, fg) in _CATEGORY_PALETTE.items()
            if cat != "unknown"
        )

        return f"""
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
            line-height:1.4">
  <div style="margin-bottom:10px;display:flex;align-items:baseline;gap:12px">
    <strong style="font-size:1.05em">{self.source_name}</strong>
    <span style="color:#888;font-size:0.88em">
      {self.shape['rows']:,} rows &times; {self.shape['columns']} columns
    </span>
  </div>

  <table style="border-collapse:collapse;width:100%;font-size:13px">
    <thead>
      <tr style="background:#f7f7f7;text-align:left;border-bottom:2px solid #ddd">
        <th style="padding:7px 10px">Column</th>
        <th style="padding:7px 10px">Dtype</th>
        <th style="padding:7px 10px">Category</th>
        <th style="padding:7px 10px">Domain</th>
        <th style="padding:7px 10px">Semantic Type</th>
        <th style="padding:7px 10px;text-align:center">Nulls</th>
        <th style="padding:7px 10px;text-align:right">Unique</th>
        <th style="padding:7px 10px">Sample Values</th>
        <th style="padding:7px 10px">Notes</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows_html)}
    </tbody>
  </table>

  <div style="margin-top:12px;padding:8px;background:#fafafa;
              border-radius:6px;border:1px solid #eee">
    <span style="font-size:0.78em;color:#999;margin-right:6px">Domains:</span>
    {legend_items}
  </div>
</div>
"""

    def __repr__(self) -> str:
        lines = [
            f"SchemaDiscoveryResult: {self.source_name}",
            f"  {self.shape['rows']:,} rows × {self.shape['columns']} columns",
            "",
            f"  {'Column':<28} {'Category':<14} {'Domain':<35} {'Nulls':>7}",
            f"  {'-'*28} {'-'*14} {'-'*35} {'-'*7}",
        ]
        for col, p in self.profile.items():
            clf = self.classifications.get(col, {})
            null_pct = f"{p.get('null_rate', 0)*100:.1f}%"
            domain = clf.get("domain", "?")
            if len(domain) > 33:
                domain = domain[:30] + "..."
            lines.append(
                f"  {col:<28} {clf.get('category','?'):<14} {domain:<35} {null_pct:>7}"
            )
        return "\n".join(lines)

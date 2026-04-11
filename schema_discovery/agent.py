"""
SchemaDiscoveryAgent: two-pass Gemini-powered schema discovery.

Pass 1 — Category detection
    Sends a lightweight profile (dtype, nulls, cardinality, samples) to Gemini.
    Gemini identifies the broad category of each column.

Pass 2 — Deep classification
    Uses the category from Pass 1 to compute only the relevant stats per column.
    Sends the enriched profile back to Gemini for the final classification:
    refined category, inferred domain, semantic type, and data quality notes.
"""
from __future__ import annotations

import json
import os
from typing import Union

import pandas as pd
from google import genai
from google.genai import types as genai_types

from .loaders import load
from .profiler import lightweight_profile_dataframe, targeted_profile
from .result import SchemaDiscoveryResult

DEFAULT_MODEL    = "gemini-2.0-flash"
DEFAULT_LOCATION = "us-central1"

# ──────────────────────────────────────────────────────────────
# Pass 1 prompt: category detection from minimal profile
# ──────────────────────────────────────────────────────────────
_PASS1_SYSTEM = """You are a data type classifier.

You will receive a minimal profile of a dataset's columns (dtype, null rate,
unique count, unique rate, and a few sample values).

Your job is to assign each column to exactly one of these 12 categories:

  numeric_continuous  — real-valued measurements (height, temperature, distance)
  numeric_discrete    — whole number counts (quantity, age, visit_count)
  numeric_ratio       — proportions/rates between 0–1 or 0–100 (accuracy, churn_rate)
  numeric_amount      — monetary or financial values (price, revenue, salary)
  temporal            — dates, times, timestamps, datetime strings
  categorical_nominal — unordered discrete labels (country, status, brand)
  categorical_ordinal — ordered discrete categories (rating, priority, education_level)
  boolean             — binary values only (true/false, yes/no, 0/1)
  textual             — free-form natural language (comments, descriptions, reviews)
  identifier          — IDs, keys, hashes — for row identification, not analysis
  geographic          — location data (lat/lon, city, zip_code, country_code)
  unknown             — cannot be determined from available information

Respond ONLY with a valid JSON array. Each element must have exactly two keys:
- "column"   : column name (string, must match exactly)
- "category" : one of the 12 category labels above (string)

Return the array and nothing else."""


# ──────────────────────────────────────────────────────────────
# Pass 2 prompt: deep classification with targeted stats
# ──────────────────────────────────────────────────────────────
_PASS2_SYSTEM = """You are an expert data analyst specializing in data schema discovery.

You will receive an enriched profile of a dataset's columns. Each column includes:
- A suggested category from a previous analysis step
- Targeted statistics computed specifically for that category
- Sample values

Your job for each column:

1. Confirm or refine the suggested category if the enriched stats reveal something different.
   Use the same 12 categories:
     numeric_continuous | numeric_discrete | numeric_ratio | numeric_amount |
     temporal | categorical_nominal | categorical_ordinal | boolean |
     textual | identifier | geographic | unknown

2. Freely infer a specific "domain" label — do NOT choose from a fixed list.
   Be as precise and descriptive as possible. Examples:
     "monthly_recurring_revenue_usd", "unix_epoch_milliseconds",
     "iso_3166_alpha2_country_code", "net_promoter_score_0_10",
     "hashed_user_fingerprint", "free_form_support_ticket_text",
     "binary_churn_event_flag", "customer_age_years", "product_sku_code"

3. Write a short "semantic_type" — a human-readable phrase for what this column represents.

4. Add "notes" for any data quality issues, anomalies, or important patterns you observe.
   Examples: high null rate, suspicious outliers, mixed formats, unexpected negatives.
   Set to null if nothing notable.

Respond ONLY with a valid JSON array. Each element must have exactly these keys:
- "column"       : column name (string, must match exactly)
- "category"     : confirmed or refined category (string)
- "domain"       : your freely inferred domain label in snake_case (string)
- "semantic_type": short human-readable description (string)
- "notes"        : data quality observations or null

Return the array and nothing else."""


class SchemaDiscoveryAgent:
    """
    Discovers the schema, data types, and column domains of a dataset using Google Gemini.

    Uses a two-pass approach:
      - Pass 1: lightweight profile → Gemini detects column categories
      - Pass 2: targeted stats per category → Gemini produces deep classification

    Authentication — two modes:

    **Vertex AI (recommended on GCP / Vertex AI Workbench)**
        Uses the VM's service account via Application Default Credentials.
        No API key required.

        >>> agent = SchemaDiscoveryAgent(project="my-gcp-project")
        >>> agent = SchemaDiscoveryAgent(project="my-gcp-project", location="europe-west4")

    **Gemini API (outside GCP)**
        Pass an API key or set the ``GEMINI_API_KEY`` environment variable.

        >>> agent = SchemaDiscoveryAgent(api_key="AIza...")

    Parameters
    ----------
    project : str, optional
        GCP project ID. When provided, uses Vertex AI (ADC auth).
        Falls back to ``GOOGLE_CLOUD_PROJECT`` or ``GCLOUD_PROJECT`` env vars.
    location : str
        Vertex AI region. Defaults to ``us-central1``.
    api_key : str, optional
        Gemini API key for non-GCP usage. Falls back to ``GEMINI_API_KEY`` env var.
    model : str
        Gemini model to use. Defaults to ``gemini-2.0-flash``.
    """

    def __init__(
        self,
        project: str | None = None,
        location: str = DEFAULT_LOCATION,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ):
        self._model = model

        resolved_project = (
            project
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCLOUD_PROJECT")
        )

        if resolved_project:
            self._client = genai.Client(
                vertexai=True,
                project=resolved_project,
                location=location,
            )
        else:
            key = api_key or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise ValueError(
                    "No authentication provided.\n"
                    "  On Vertex AI Workbench: pass project='your-gcp-project'\n"
                    "  Outside GCP: pass api_key= or set GEMINI_API_KEY env var"
                )
            self._client = genai.Client(api_key=key)

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def discover(
        self,
        source: Union[str, pd.DataFrame],
        name: str | None = None,
        sample_rows: int = 5_000,
        **load_kwargs,
    ) -> SchemaDiscoveryResult:
        """
        Run two-pass schema discovery on a file path or DataFrame.

        Parameters
        ----------
        source : str | pd.DataFrame
            File path (CSV, Parquet, Excel, …) or an existing DataFrame.
        name : str, optional
            Display name for the dataset. Inferred from file path if not provided.
        sample_rows : int
            Maximum rows to profile. Random sample taken if dataset is larger.
        **load_kwargs
            Extra arguments forwarded to the file loader
            (e.g. sep="|" for CSV, sheet_name="Sales" for Excel).

        Returns
        -------
        SchemaDiscoveryResult
        """
        # ── Load ──────────────────────────────────────────────
        if isinstance(source, str):
            df = load(source, **load_kwargs)
            source_name = name or source
        elif isinstance(source, pd.DataFrame):
            df = source.copy()
            source_name = name or "DataFrame"
        else:
            raise TypeError(
                f"source must be a file path (str) or pd.DataFrame, got {type(source)}"
            )

        if len(df) > sample_rows:
            df = df.sample(n=sample_rows, random_state=42).reset_index(drop=True)

        # ── Pass 1: lightweight profile → category detection ──
        light = lightweight_profile_dataframe(df)
        pass1_prompt = self._build_pass1_prompt(light, source_name)
        categories = self._gemini_call(pass1_prompt, _PASS1_SYSTEM)
        category_map = {row["column"]: row["category"] for row in categories}

        # ── Pass 2: targeted stats → deep classification ───────
        enriched = self._build_enriched_profile(df, light["columns"], category_map)
        pass2_prompt = self._build_pass2_prompt(enriched, light["shape"], source_name)
        classifications = self._gemini_call(pass2_prompt, _PASS2_SYSTEM)

        return SchemaDiscoveryResult(
            source_name=source_name,
            shape=light["shape"],
            profile=light["columns"],
            classifications=classifications,
        )

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _build_pass1_prompt(self, light: dict, dataset_name: str) -> str:
        lines = [
            f"Dataset: {dataset_name}",
            f"Shape: {light['shape']['rows']:,} rows × {light['shape']['columns']} columns",
            "",
            "Column profiles (minimal):",
        ]
        for col, p in light["columns"].items():
            lines.append(f'\n  Column: "{col}"')
            lines.append(f"  {json.dumps(p, default=str)}")
        return "\n".join(lines)

    def _build_enriched_profile(
        self,
        df: pd.DataFrame,
        light_columns: dict,
        category_map: dict,
    ) -> dict:
        """
        For each column, merge the lightweight profile with targeted stats
        computed based on the category Gemini assigned in Pass 1.
        """
        enriched = {}
        for col in df.columns:
            category = category_map.get(col, "unknown")
            extra = targeted_profile(df[col], category)
            enriched[col] = {
                "suggested_category": category,
                **light_columns[col],
                **extra,
            }
        return enriched

    def _build_pass2_prompt(self, enriched: dict, shape: dict, dataset_name: str) -> str:
        lines = [
            f"Dataset: {dataset_name}",
            f"Shape: {shape['rows']:,} rows × {shape['columns']} columns",
            "",
            "Enriched column profiles:",
        ]
        for col, p in enriched.items():
            exclude = {"sample_values"}
            stats = {k: v for k, v in p.items() if k not in exclude and v is not None}
            samples = p.get("sample_values", [])
            lines.append(f'\n  Column: "{col}"')
            lines.append(f"  Profile: {json.dumps(stats, default=str)}")
            lines.append(f"  Sample values: {samples}")
        return "\n".join(lines)

    def _gemini_call(self, prompt: str, system_prompt: str) -> list[dict]:
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Gemini returned invalid JSON.\nRaw response:\n{raw}"
            ) from exc
        if not isinstance(result, list):
            raise ValueError(
                f"Expected a JSON array from Gemini, got {type(result).__name__}"
            )
        return result

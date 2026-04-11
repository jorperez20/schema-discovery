"""
SchemaDiscoveryAgent: orchestrates data loading, profiling, and Gemini-powered classification.
"""
from __future__ import annotations

import json
import os
from typing import Union

import pandas as pd
from google import genai
from google.genai import types as genai_types

from .loaders import load
from .profiler import profile_dataframe
from .classifier import pre_classify
from .result import SchemaDiscoveryResult

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_LOCATION = "us-central1"

_SYSTEM_PROMPT = """You are an expert data analyst specializing in data schema discovery.

You will receive a statistical profile of a dataset's columns. For each column you must:

1. Freely infer a specific "domain" label — do NOT choose from a fixed list.
   The domain should be as precise and descriptive as possible, capturing what the data
   actually represents. Examples of good domain labels:
     "monthly_recurring_revenue_usd", "unix_epoch_milliseconds", "iso_3166_alpha2_country_code",
     "net_promoter_score_0_10", "hashed_user_fingerprint", "free_form_support_ticket_text",
     "binary_churn_event_flag", "customer_age_years", "product_sku_code",
     "latitude_wgs84", "email_address", "http_status_code", "boolean_is_active"

2. Assign a "category" — one of exactly these 6 broad buckets:
     - numeric      : any number intended for calculation (amounts, scores, counts, ratios,
                      rates, durations in seconds, measurements)
     - temporal     : dates, times, timestamps, datetime strings
     - categorical  : discrete labels, boolean flags, ordinal levels, status fields,
                      enumerations — anything with a finite set of meaningful values
     - textual      : free-form natural language (comments, descriptions, reviews, names)
     - identifier   : IDs, keys, codes, hashes — used for row identification, not analysis
     - geographic   : location data (coordinates, country codes, city names, postal codes)

Respond ONLY with a valid JSON array. Each element must have exactly these keys:
- "column"       : column name (string, must match exactly)
- "category"     : one of the 6 category labels above (string)
- "domain"       : your freely inferred specific domain label (string, snake_case)
- "semantic_type": a short human-readable phrase describing what this column represents
- "notes"        : data quality observations, anomalies, or important patterns — or null

Use the statistical hints as soft signals, but apply your own domain expertise.
Return the JSON array and nothing else."""


class SchemaDiscoveryAgent:
    """
    Discovers the schema, data types, and column domains of a dataset using Google Gemini.

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
        GCP project ID. When provided, uses Vertex AI (ADC auth). Takes priority over api_key.
        Falls back to the ``GOOGLE_CLOUD_PROJECT`` or ``GCLOUD_PROJECT`` environment variable.
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

        # Resolve GCP project from env if not passed explicitly
        resolved_project = (
            project
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCLOUD_PROJECT")
        )

        if resolved_project:
            # Vertex AI mode — ADC handles auth automatically on GCP VMs
            self._client = genai.Client(
                vertexai=True,
                project=resolved_project,
                location=location,
            )
        else:
            # Gemini API mode — requires an API key
            key = api_key or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise ValueError(
                    "No authentication provided.\n"
                    "  On Vertex AI Workbench: pass project='your-gcp-project'\n"
                    "  Outside GCP: pass api_key= or set GEMINI_API_KEY env var"
                )
            self._client = genai.Client(api_key=key)

    def discover(
        self,
        source: Union[str, pd.DataFrame],
        name: str | None = None,
        sample_rows: int = 5_000,
        **load_kwargs,
    ) -> SchemaDiscoveryResult:
        """
        Run schema discovery on a file path or DataFrame.

        Parameters
        ----------
        source : str | pd.DataFrame
            File path (CSV, Parquet, Excel, …) or an existing DataFrame.
        name : str, optional
            Display name for the dataset. Inferred from file path if not provided.
        sample_rows : int
            Maximum rows to profile (random sample if dataset is larger).
        **load_kwargs
            Additional keyword arguments forwarded to the file loader
            (e.g. ``sep="|"`` for CSV, ``sheet_name="Sheet2"`` for Excel).

        Returns
        -------
        SchemaDiscoveryResult
        """
        # --- Load ---
        if isinstance(source, str):
            df = load(source, **load_kwargs)
            source_name = name or source
        elif isinstance(source, pd.DataFrame):
            df = source.copy()
            source_name = name or "DataFrame"
        else:
            raise TypeError(f"source must be a file path (str) or pd.DataFrame, got {type(source)}")

        # --- Sample ---
        if len(df) > sample_rows:
            df = df.sample(n=sample_rows, random_state=42).reset_index(drop=True)

        # --- Profile ---
        profile = profile_dataframe(df)

        # --- Pre-classify (rule-based hints) ---
        hints = {
            col: pre_classify(col, p)
            for col, p in profile["columns"].items()
        }

        # --- Classify with Gemini ---
        prompt = self._build_prompt(profile, hints, source_name)
        classifications = self._call_gemini(prompt)

        return SchemaDiscoveryResult(
            source_name=source_name,
            shape=profile["shape"],
            profile=profile["columns"],
            classifications=classifications,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, profile: dict, hints: dict, dataset_name: str) -> str:
        lines = [
            f"Dataset: {dataset_name}",
            f"Shape: {profile['shape']['rows']:,} rows × {profile['shape']['columns']} columns",
            "",
            "Column profiles:",
        ]
        for col, p in profile["columns"].items():
            hint = hints.get(col)
            hint_str = f"  [hint: {hint}]" if hint else ""
            # Exclude verbose/redundant keys for the prompt
            exclude = {"sample_values", "looks_like_dates"}
            stats = {k: v for k, v in p.items() if k not in exclude and v is not None}
            samples = p.get("sample_values", [])
            lines.append(f'\n  Column: "{col}"{hint_str}')
            lines.append(f"  Stats:  {json.dumps(stats, default=str)}")
            lines.append(f"  Sample values: {samples}")
        return "\n".join(lines)

    def _call_gemini(self, prompt: str) -> list[dict]:
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        raw = response.text.strip()
        # Strip markdown fences if the model wraps the output
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0].strip()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Gemini returned invalid JSON.\nRaw response:\n{raw}"
            ) from exc
        if not isinstance(result, list):
            raise ValueError(
                f"Expected a JSON array from Gemini, got: {type(result).__name__}"
            )
        return result

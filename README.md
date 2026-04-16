# Schema Discovery

AI-powered tool that automatically analyzes dataset structure and infers semantic information about columns using Google Gemini. Identifies column categories, infers domain-specific labels, and flags data quality issues — no manual annotation required.

## How it works

Uses a **two-pass pipeline** optimized for cost and accuracy:

1. **Pass 1 — lightweight profile:** Computes minimal stats (dtype, nulls, cardinality, samples) and sends to Gemini to detect broad column categories.
2. **Pass 2 — targeted profiling:** Based on detected categories, computes only relevant stats (e.g., date ranges for temporal, value counts for categorical) and sends the enriched profile to Gemini for deep classification.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+.

## Authentication

**Vertex AI (GCP)** — uses Application Default Credentials:

```python
agent = SchemaDiscoveryAgent(project="my-gcp-project")
```

**Gemini API** — uses an API key:

```python
agent = SchemaDiscoveryAgent(api_key="AIza...")
# or set the GEMINI_API_KEY environment variable
```

## Usage

```python
from schema_discovery import SchemaDiscoveryAgent

agent = SchemaDiscoveryAgent(project="my-gcp-project")

# From a file (CSV, Parquet, Excel, JSON, TSV)
result = agent.discover("data/my_dataset.csv")

# From a pandas DataFrame
result = agent.discover(df, name="my_dataset")

# View results
print(result)                        # summary table
result.to_dataframe()                # pandas DataFrame
result.filter(category="temporal")  # filter by category or domain
```

Results render as styled HTML tables in Jupyter notebooks.

## Column categories

| Category | Description |
|---|---|
| `numeric_continuous` | Continuous numeric measurements |
| `numeric_discrete` | Discrete counts or integer values |
| `numeric_ratio` | Ratios or proportions (0–1) |
| `numeric_amount` | Currency or monetary amounts |
| `temporal` | Dates, times, timestamps |
| `categorical_nominal` | Unordered categorical values |
| `categorical_ordinal` | Ordered categorical values |
| `boolean` | Binary / flag columns |
| `textual` | Free-form text |
| `identifier` | IDs, keys, codes |
| `geographic` | Location-related values |
| `unknown` | Unclassified |

## Dependencies

| Package | Purpose |
|---|---|
| `google-genai` | Gemini API client |
| `google-auth` | GCP authentication |
| `pandas` | DataFrames |
| `numpy` | Numerical operations |
| `openpyxl` / `xlrd` | Excel support |
| `pyarrow` | Parquet support |

## Default model

`gemini-2.5-pro` (configurable via the `model` parameter on `SchemaDiscoveryAgent`).

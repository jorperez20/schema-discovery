"""
Extensible file loader registry.
Supports CSV, Parquet, Excel out of the box.
Register additional formats with `register()`.
"""
import pandas as pd
from pathlib import Path
from typing import Callable

_LOADERS: dict[str, Callable[..., pd.DataFrame]] = {}


def _loader(extension: str):
    def decorator(fn):
        _LOADERS[extension.lower()] = fn
        return fn
    return decorator


@_loader(".csv")
def _load_csv(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


@_loader(".parquet")
def _load_parquet(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_parquet(path, **kwargs)


@_loader(".xlsx")
def _load_xlsx(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_excel(path, **kwargs)


@_loader(".xls")
def _load_xls(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_excel(path, engine="xlrd", **kwargs)


@_loader(".json")
def _load_json(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_json(path, **kwargs)


@_loader(".tsv")
def _load_tsv(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", **kwargs)


def load(path: str, **kwargs) -> pd.DataFrame:
    """Load a file into a DataFrame. Extension determines the loader."""
    ext = Path(path).suffix.lower()
    if ext not in _LOADERS:
        supported = sorted(_LOADERS.keys())
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Supported: {supported}. "
            f"Register custom formats with schema_discovery.loaders.register()."
        )
    return _LOADERS[ext](path, **kwargs)


def register(extension: str, loader_fn: Callable[..., pd.DataFrame]) -> None:
    """
    Register a custom loader for a file extension.

    Example:
        from schema_discovery.loaders import register
        register(".feather", lambda path, **kw: pd.read_feather(path, **kw))
    """
    ext = extension if extension.startswith(".") else f".{extension}"
    _LOADERS[ext.lower()] = loader_fn


def supported_formats() -> list[str]:
    """Return list of currently registered file extensions."""
    return sorted(_LOADERS.keys())

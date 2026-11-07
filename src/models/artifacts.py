"""Save and load build artifacts.

Layout (all git-ignored):
    Data/2_Processed/<dataset>.parquet       processed analysis DataFrames
    artifacts/results/<bundle>/<table>.parquet
    artifacts/results/<bundle>/meta.json     scalars, dicts, small lists
    artifacts/models/<bundle>/<model>.joblib fitted estimators
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR


class ArtifactsNotBuiltError(FileNotFoundError):
    """Raised when the build script has not been run yet."""


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna()
            if len(sample) and sample.map(type).nunique() > 1:
                df[col] = df[col].astype("string")
    return df


# ---------------------------------------------------------------------------
# Processed data
# ---------------------------------------------------------------------------
def save_processed(frames: dict[str, pd.DataFrame]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        _parquet_safe(df).to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)


def processed_available(names: list[str]) -> bool:
    return all((PROCESSED_DIR / f"{n}.parquet").exists() for n in names)


def load_processed(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise ArtifactsNotBuiltError(str(path))
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Result bundles
# ---------------------------------------------------------------------------
def save_bundle(bundle: str, tables: dict[str, pd.DataFrame], meta: dict[str, Any]) -> Path:
    folder = RESULTS_DIR / bundle
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        index_name = frame.index.name
        if index_name is not None or not isinstance(frame.index, pd.RangeIndex):
            frame = frame.reset_index()
        _parquet_safe(frame).to_parquet(folder / f"{name}.parquet", index=False)
    with open(folder / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=_json_default)
    return folder


def bundle_available(bundle: str) -> bool:
    return (RESULTS_DIR / bundle / "meta.json").exists()


def load_bundle(bundle: str) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    folder = RESULTS_DIR / bundle
    if not (folder / "meta.json").exists():
        raise ArtifactsNotBuiltError(
            f"Results '{bundle}' not found. Run: python scripts/build_artifacts.py")
    tables = {p.stem: pd.read_parquet(p) for p in sorted(folder.glob("*.parquet"))}
    with open(folder / "meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    return tables, meta


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def save_models(bundle: str, models: dict[str, Any]) -> dict[str, str]:
    folder = MODELS_DIR / bundle
    folder.mkdir(parents=True, exist_ok=True)
    index = {}
    for name, model in models.items():
        file = f"{_slug(name)}.joblib"
        joblib.dump(model, folder / file, compress=3)
        index[name] = file
    with open(folder / "index.json", "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    return index


def load_model(bundle: str, name: str) -> Any:
    folder = MODELS_DIR / bundle
    index_path = folder / "index.json"
    if not index_path.exists():
        raise ArtifactsNotBuiltError(f"Models '{bundle}' not found. Run scripts/build_artifacts.py")
    with open(index_path, encoding="utf-8") as fh:
        index = json.load(fh)
    return joblib.load(folder / index[name])

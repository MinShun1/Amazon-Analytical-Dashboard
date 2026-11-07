"""Streamlit-cached access to processed data and build artifacts.

Pages import from here only. Processed parquet files are preferred; if they
are missing, the pipeline runs once from the raw zip (a few seconds), so the
descriptive pages work even before the model artifacts have been built.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.data.loader import check_raw_zip, read_raw
from src.data.preprocessing import build_analysis_data
from src.data.validation import load_reference
from src.models import artifacts as art

DATASETS = ["amazon", "international", "international_full", "may_2022", "pnl_2021",
            "product", "received_amount", "expense_detail", "warehouse_pricing"]


@st.cache_data(show_spinner="Preparing data from the raw zip ...")
def _pipeline_frames() -> dict[str, pd.DataFrame]:
    data = build_analysis_data(read_raw()).as_dict()
    return {k: v for k, v in data.items() if k in DATASETS}


@st.cache_data(show_spinner=False)
def dataset(name: str) -> pd.DataFrame:
    if art.processed_available([name]):
        return art.load_processed(name)
    return _pipeline_frames()[name]


def data_problem() -> str | None:
    """A user-facing message if no data source is available, else None."""
    if art.processed_available(DATASETS):
        return None
    problems = check_raw_zip()
    return " ".join(problems) if problems else None


@st.cache_data(show_spinner=False)
def results(bundle: str) -> tuple[dict[str, pd.DataFrame], dict[str, Any]] | None:
    if not art.bundle_available(bundle):
        return None
    return art.load_bundle(bundle)


@st.cache_data(show_spinner=False)
def reference() -> dict[str, Any]:
    return load_reference()


@st.cache_resource(show_spinner="Loading model ...")
def model(bundle: str, name: str):
    return art.load_model(bundle, name)

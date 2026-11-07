"""Reproduce the notebook's *effective* preprocessing for the eight DataFrames.

Each function mirrors the cells that actually shaped the data the notebook's
analyses used (cell numbers in comments). Notable behaviours are preserved on
purpose, not "fixed" (see README > Methodology notes):

* Amazon: cell 108 re-assigned ``amazon = dfs["amazon"]``, which discarded the
  earlier snake_case/dtype cleaning (cells 16-19) and the ``amount=0`` fill
  (cell 57). All later analyses use the raw column names, so we do the same.
* International: only the first 18,635 rows form the main table (cell 62);
  analyses use a candidate-key de-duplicated copy (cell 164).
* No missing values are imputed and no cross-dataset joins are made.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import (
    INTERNATIONAL_KEY_COLS,
    INTERNATIONAL_MAIN_TABLE_ROWS,
)


def _snake_columns(df: pd.DataFrame, dash: bool = False) -> pd.DataFrame:
    cols = df.columns.str.strip().str.lower().str.replace(" ", "_")
    if dash:
        cols = cols.str.replace("-", "_")
    df.columns = cols
    return df


def _clean_strings(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().replace("", pd.NA)


def add_sku_standardized(df: pd.DataFrame, sku_col: str = "sku") -> pd.DataFrame:
    """Cell 149: per-dataset SKU standardisation (never used as a join key)."""
    df["sku_standardized"] = (
        df[sku_col].astype("string").str.strip().str.upper().replace("", pd.NA)
    )
    return df


# ---------------------------------------------------------------------------
# Amazon
# ---------------------------------------------------------------------------
def clean_amazon(raw: pd.DataFrame) -> pd.DataFrame:
    """Effective Amazon state (cells 108-115, 149). Raw column names kept."""
    amazon = raw.copy()
    amazon["ship-state"] = amazon["ship-state"].str.strip()            # 110
    amazon = amazon.drop(columns=["Unnamed: 22"])                      # 112
    amazon["Status"] = amazon["Status"].str.strip()
    for col in ["ship-city", "ship-state"]:                            # 114
        amazon[col] = amazon[col].astype("string").str.strip().str.upper()
    amazon["promotion-ids"] = amazon["promotion-ids"].str.strip()      # 115
    amazon = amazon.rename(columns={"SKU": "sku"})                     # 149
    return add_sku_standardized(amazon)


def amazon_analysis_copy(amazon: pd.DataFrame) -> pd.DataFrame:
    """Cell 168: analysis copy with ``Date`` parsed using the validated format."""
    amz = amazon.copy()
    amz["Date"] = pd.to_datetime(amz["Date"], format="%m-%d-%y", errors="coerce")
    return amz


# ---------------------------------------------------------------------------
# International
# ---------------------------------------------------------------------------
def clean_international(raw: pd.DataFrame) -> pd.DataFrame:
    """Cells 25, 62, 120, 149 (before de-duplication)."""
    intl = raw.copy().drop(columns=["index"])
    intl = _snake_columns(intl)
    intl["date"] = pd.to_datetime(intl["date"], format="%m-%d-%y", errors="coerce")
    for col in ["pcs", "rate", "gross_amt"]:
        intl[col] = pd.to_numeric(intl[col], errors="coerce")

    intl = intl.iloc[:INTERNATIONAL_MAIN_TABLE_ROWS].copy()           # 62

    for col in ["months", "customer", "style", "sku", "size"]:         # 120
        intl[col] = _clean_strings(intl[col])
    intl["size"] = intl["size"].replace({
        "s": "S", "m": "M", "l": "L", "xl": "XL", "xxl": "XXL",
        "xxxl": "XXXL", "FREE": "Free",
    })
    return add_sku_standardized(intl)                                  # 149


def deduplicate_international(intl: pd.DataFrame) -> pd.DataFrame:
    """Cell 164: drop repeated *complete* candidate keys, keep first; rows
    with an incomplete key are retained because identity can't be established."""
    complete = intl[INTERNATIONAL_KEY_COLS].notna().all(axis=1)
    complete_dedup = intl.loc[complete].drop_duplicates(
        subset=INTERNATIONAL_KEY_COLS, keep="first"
    )
    return pd.concat([complete_dedup, intl.loc[~complete]]).sort_index()


# ---------------------------------------------------------------------------
# Pricing datasets
# ---------------------------------------------------------------------------
_PRICE_STRING_COLS = ["sku", "style_id", "catalog", "category"]


def _clean_pricing(raw: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    df = raw.copy().drop(columns=["index"])
    df = _snake_columns(df)
    for col in _PRICE_STRING_COLS:
        df[col] = df[col].astype("string").str.strip()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_may_2022(raw: pd.DataFrame) -> pd.DataFrame:
    """Cells 29, 130, 149."""
    numeric = ["weight", "tp", "mrp_old", "final_mrp_old"] + [
        "ajio_mrp", "amazon_mrp", "amazon_fba_mrp", "flipkart_mrp",
        "limeroad_mrp", "myntra_mrp", "paytm_mrp", "snapdeal_mrp",
    ]
    df = _clean_pricing(raw, numeric)
    for col in _PRICE_STRING_COLS:
        df[col] = _clean_strings(df[col])
    return add_sku_standardized(df)


def clean_pnl_2021(raw: pd.DataFrame) -> pd.DataFrame:
    """Cells 33, 125, 149."""
    numeric = ["weight", "tp_1", "tp_2", "mrp_old", "final_mrp_old"] + [
        "ajio_mrp", "amazon_mrp", "amazon_fba_mrp", "flipkart_mrp",
        "limeroad_mrp", "myntra_mrp", "paytm_mrp", "snapdeal_mrp",
    ]
    df = _clean_pricing(raw, numeric)
    for col in _PRICE_STRING_COLS:
        df[col] = df[col].astype("string").str.strip()
    return add_sku_standardized(df)


# ---------------------------------------------------------------------------
# Product inventory
# ---------------------------------------------------------------------------
def clean_product(raw: pd.DataFrame) -> pd.DataFrame:
    """Cells 36, 85, 89, 91, 136, 149."""
    product = raw.copy().drop(columns=["index"])
    product = _snake_columns(product)
    for col in ["sku_code", "design_no.", "category", "size", "color"]:
        product[col] = product[col].astype("string").str.strip()
    product["stock"] = pd.to_numeric(product["stock"], errors="coerce")

    product = product.dropna(how="all").reset_index(drop=True)            # 85
    product["design_no."] = product["design_no."].replace("#REF!", pd.NA)  # 89
    product = product.drop_duplicates().reset_index(drop=True)            # 91

    for col in ["sku_code", "design_no.", "category", "size", "color"]:   # 136
        product[col] = _clean_strings(product[col])
    product["sku_code"] = product["sku_code"].replace("#REF!", pd.NA)
    product["category"] = product["category"].str.upper()
    product["color"] = product["color"].str.upper()

    product = product.rename(columns={"sku_code": "sku"})                 # 149
    return add_sku_standardized(product)


# ---------------------------------------------------------------------------
# Financial & operational
# ---------------------------------------------------------------------------
def split_expense_file(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cell 11 + 40/97 (received) and 43/101/142 (expense)."""
    expense_raw = raw.copy().drop(columns=["index"])

    received = expense_raw[["Recived Amount", "Unnamed: 1"]].copy()
    received.columns = ["date", "received_amount"]
    received = received.iloc[1:].copy()
    received = received[received["date"].notna()].reset_index(drop=True)
    received["date"] = pd.to_datetime(received["date"], format="%m-%d-%y", errors="coerce")
    received["received_amount"] = pd.to_numeric(received["received_amount"], errors="coerce")
    received = received.dropna(subset=["date"]).reset_index(drop=True)   # 97: drop total row

    expense = expense_raw[["Expance", "Unnamed: 3"]].copy()
    expense.columns = ["expense_item", "expense_amount"]
    expense = expense.iloc[1:].copy()
    expense = expense[
        expense["expense_item"].notna() | expense["expense_amount"].notna()
    ].reset_index(drop=True)
    expense["expense_item"] = expense["expense_item"].astype("string").str.strip()
    expense["expense_amount"] = pd.to_numeric(expense["expense_amount"], errors="coerce")
    expense = expense.drop(index=13).reset_index(drop=True)              # 101: 8,095 subtotal
    expense["expense_item"] = _clean_strings(expense["expense_item"])    # 142
    return received, expense


def clean_warehouse_pricing(raw: pd.DataFrame) -> pd.DataFrame:
    """Cells 10, 47, 48, 146, 168. Cell 47's positional relabel is the
    correct final labelling (descriptions sit in the raw 'Shiprocket' column)."""
    wh = raw.copy().drop(columns=["index"])
    wh = wh.iloc[1:5].copy()
    wh = wh.rename(columns={
        "Shiprocket": "shiprocket_price", "Unnamed: 1": "head", "INCREFF": "increff_price",
    })
    wh = wh[["head", "shiprocket_price", "increff_price"]].reset_index(drop=True)
    wh.columns = ["shiprocket_price", "head", "increff_price"]           # 47

    wh["head"] = _clean_strings(wh["head"])                              # 48 / 146
    wh["shiprocket_price"] = pd.to_numeric(
        wh["shiprocket_price"].astype("string").str.replace("₹", "", regex=False).str.strip(),
        errors="coerce",
    )
    wh["increff_price"] = wh["increff_price"].astype("string").str.strip()
    wh["increff_price_numeric"] = pd.to_numeric(wh["increff_price"], errors="coerce")  # 168
    return wh


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass
class AnalysisData:
    """The eight analysis DataFrames (``analysis_data`` in the notebook),
    plus the pre-dedup International table for transparency."""

    amazon: pd.DataFrame            # Date parsed (analysis copy)
    amazon_source: pd.DataFrame     # Date as raw string (``amazon`` variable)
    international: pd.DataFrame     # de-duplicated (``international_dedup``)
    international_full: pd.DataFrame
    may_2022: pd.DataFrame
    pnl_2021: pd.DataFrame
    product: pd.DataFrame
    received_amount: pd.DataFrame
    expense_detail: pd.DataFrame
    warehouse_pricing: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return dict(self.__dict__)


def build_analysis_data(raw: dict[str, pd.DataFrame]) -> AnalysisData:
    amazon_source = clean_amazon(raw["amazon"])
    intl_full = clean_international(raw["international"])
    received, expense = split_expense_file(raw["expense"])
    return AnalysisData(
        amazon=amazon_analysis_copy(amazon_source),
        amazon_source=amazon_source,
        international=deduplicate_international(intl_full),
        international_full=intl_full,
        may_2022=clean_may_2022(raw["may_2022"]),
        pnl_2021=clean_pnl_2021(raw["pnl_2021"]),
        product=clean_product(raw["product"]),
        received_amount=received,
        expense_detail=expense,
        warehouse_pricing=clean_warehouse_pricing(raw["warehouse"]),
    )

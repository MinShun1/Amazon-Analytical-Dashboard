"""Reconcile reproduced values against the notebook's saved outputs."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

import pandas as pd

from src.config import REFERENCE_VALUES
from src.data.preprocessing import AnalysisData


@lru_cache(maxsize=1)
def load_reference() -> dict[str, Any]:
    with open(REFERENCE_VALUES, encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class Check:
    area: str
    name: str
    expected: Any
    actual: Any
    tolerance: float = 0.0

    @property
    def ok(self) -> bool:
        if isinstance(self.expected, (int, float)) and isinstance(self.actual, (int, float)):
            if math.isnan(float(self.actual)):
                return False
            return abs(float(self.expected) - float(self.actual)) <= self.tolerance
        return self.expected == self.actual

    def as_row(self) -> dict[str, Any]:
        return {
            "area": self.area, "check": self.name, "notebook": self.expected,
            "reproduced": self.actual, "tolerance": self.tolerance,
            "status": "OK" if self.ok else "DIFF",
        }


def to_frame(checks: Iterable[Check]) -> pd.DataFrame:
    return pd.DataFrame([c.as_row() for c in checks])


METRIC_COLS = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]


def metrics_vs_notebook(reproduced: pd.DataFrame, notebook: dict[str, dict[str, float]],
                        decimals: int = 4) -> pd.DataFrame:
    """Reproduced metrics with the notebook's printed values side by side.

    ``max_abs_diff`` is measured after rounding to the notebook's precision;
    ``matches_notebook`` is True when every metric agrees at that precision.
    """
    out = reproduced.copy()
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    diffs = []
    for _, row in out.iterrows():
        ref = notebook.get(row["Model"], {})
        diffs.append(max((abs(row[m] - ref[m]) for m in METRIC_COLS if m in ref), default=float("nan")))
        for m in METRIC_COLS:
            out.loc[out["Model"] == row["Model"], f"Notebook {m}"] = ref.get(m)
    out["max_abs_diff"] = diffs
    out["matches_notebook"] = out["max_abs_diff"] <= tol
    return out


def dataset_checks(data: AnalysisData) -> list[Check]:
    ref = load_reference()["datasets"]
    a, i, p = data.amazon, data.international, data.product
    return [
        Check("amazon", "rows", ref["amazon"]["rows"], len(a)),
        Check("amazon", "unique orders", ref["amazon"]["unique_orders"], a["Order ID"].nunique()),
        Check("amazon", "units", ref["amazon"]["units"], int(a["Qty"].sum())),
        Check("amazon", "recorded amount", ref["amazon"]["recorded_amount"],
              round(float(a["Amount"].sum(min_count=1)), 2), 0.01),
        Check("amazon", "missing amount", ref["amazon"]["missing_amount"], int(a["Amount"].isna().sum())),
        Check("amazon", "unique dates", ref["amazon"]["unique_dates"], a["Date"].nunique()),
        Check("amazon", "date min", ref["amazon"]["date_min"], str(a["Date"].min().date())),
        Check("amazon", "date max", ref["amazon"]["date_max"], str(a["Date"].max().date())),
        Check("international", "rows before dedup", ref["international_full"]["rows"], len(data.international_full)),
        Check("international", "rows after dedup", ref["international"]["rows"], len(i)),
        Check("international", "unique customers", ref["international"]["unique_customers"], i["customer"].nunique()),
        Check("international", "pieces", ref["international"]["pieces"], float(i["pcs"].sum())),
        Check("international", "gross amount", ref["international"]["gross_amount"],
              round(float(i["gross_amt"].sum()), 2), 0.01),
        Check("international", "missing sku", ref["international"]["missing_sku"], int(i["sku"].isna().sum())),
        Check("may_2022", "rows", ref["may_2022"]["rows"], len(data.may_2022)),
        Check("pnl_2021", "rows", ref["pnl_2021"]["rows"], len(data.pnl_2021)),
        Check("product", "rows", ref["product"]["rows"], len(p)),
        Check("product", "unique sku", ref["product"]["unique_sku"], p["sku"].nunique()),
        Check("product", "missing sku", ref["product"]["missing_sku"], int(p["sku"].isna().sum())),
        Check("product", "recorded stock", ref["product"]["recorded_stock"], float(p["stock"].sum())),
        Check("product", "zero-stock rows", ref["product"]["zero_stock_rows"], int(p["stock"].eq(0).sum())),
        Check("received_amount", "rows", ref["received_amount"]["rows"], len(data.received_amount)),
        Check("received_amount", "total", ref["received_amount"]["total"],
              float(data.received_amount["received_amount"].sum())),
        Check("expense_detail", "rows", ref["expense_detail"]["rows"], len(data.expense_detail)),
        Check("warehouse_pricing", "shiprocket prices", ref["warehouse_pricing"]["shiprocket_price"],
              data.warehouse_pricing["shiprocket_price"].tolist()),
    ]

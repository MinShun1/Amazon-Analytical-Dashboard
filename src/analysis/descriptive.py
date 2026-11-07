"""Descriptive KPIs and aggregations (notebook cells 171-195, 327-341).

All functions are pure: they take a (possibly filtered) DataFrame and return
a summary. Missing values are kept and labelled, never silently dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MARKETPLACE_COLS

MISSING_LABEL = "Missing"


def _sum(s: pd.Series) -> float:
    return s.sum(min_count=1)


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": df.isna().mean() * 100,
    })
    return out[out["missing_count"] > 0].sort_values("missing_count", ascending=False)


# ===========================================================================
# Amazon (raw column names, see preprocessing docstring)
# ===========================================================================
def amazon_kpis(amz: pd.DataFrame) -> dict[str, float]:
    orders = amz["Order ID"].nunique()
    return {
        "records": len(amz),
        "unique_orders": orders,
        "units": amz["Qty"].sum(),
        "recorded_amount": _sum(amz["Amount"]),
        "missing_amount": int(amz["Amount"].isna().sum()),
        "unique_dates": amz["Date"].nunique(),
        "cancelled_record_pct": amz["Status"].eq("Cancelled").mean() * 100 if len(amz) else np.nan,
        "avg_amount_per_order": _sum(amz["Amount"]) / orders if orders else np.nan,
    }


def amazon_daily(amz: pd.DataFrame) -> pd.DataFrame:
    """Cell 172."""
    return (
        amz.groupby("Date")
        .agg(
            unique_orders=("Order ID", "nunique"),
            units_recorded=("Qty", "sum"),
            recorded_amount=("Amount", _sum),
            rows=("Order ID", "size"),
            rows_missing_amount=("Amount", lambda s: s.isna().sum()),
        )
        .reset_index()
        .sort_values("Date")
    )


def amazon_monthly(amz: pd.DataFrame) -> pd.DataFrame:
    """Cell 182 (March and June are partial months)."""
    df = amz.assign(month=amz["Date"].dt.to_period("M").dt.to_timestamp())
    out = (
        df.groupby("month")
        .agg(
            records=("Order ID", "size"),
            unique_orders=("Order ID", "nunique"),
            units_recorded=("Qty", "sum"),
            recorded_amount=("Amount", _sum),
            cancelled_records=("Status", lambda s: s.eq("Cancelled").sum()),
            missing_amount_records=("Amount", lambda s: s.isna().sum()),
        )
        .reset_index()
        .sort_values("month")
    )
    out["cancelled_record_pct"] = out["cancelled_records"] / out["records"] * 100
    return out


def value_share(df: pd.DataFrame, col: str, label: str | None = None) -> pd.DataFrame:
    """Record count and share per category, missing values labelled (cell 180)."""
    label = label or col
    out = (
        df[col].astype("object").fillna(MISSING_LABEL)
        .value_counts()
        .rename_axis(label)
        .reset_index(name="records")
    )
    out["pct"] = out["records"] / len(df) * 100 if len(df) else np.nan
    return out


def amazon_breakdown(amz: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """Generic Amazon breakdown used by cells 182-183 (category, channel,
    courier status, state, size, fulfilment)."""
    out = (
        amz.groupby(by, dropna=False)
        .agg(
            records=("Order ID", "size"),
            unique_orders=("Order ID", "nunique"),
            units_recorded=("Qty", "sum"),
            recorded_amount=("Amount", _sum),
            median_amount=("Amount", "median"),
            cancelled_records=("Status", lambda s: s.eq("Cancelled").sum()),
            missing_amount_records=("Amount", lambda s: s.isna().sum()),
        )
        .reset_index()
    )
    out["record_pct"] = out["records"] / len(amz) * 100 if len(amz) else np.nan
    out["cancelled_pct"] = out["cancelled_records"] / out["records"] * 100
    cols = [by] if isinstance(by, str) else by
    for c in cols:
        out[c] = out[c].astype("object").fillna(MISSING_LABEL)
    return out


def order_level_cancellation(amz: pd.DataFrame) -> pd.DataFrame:
    """Cell 184: one row per order, cancellation rate by fulfilment."""
    order_level = (
        amz.groupby("Order ID", dropna=False)
        .agg(
            fulfilment=("Fulfilment", "first"),
            status=("Status", "first"),
            record_count=("Status", "size"),
            recorded_amount=("Amount", _sum),
        )
        .reset_index()
    )
    order_level["is_cancelled"] = order_level["status"].eq("Cancelled")
    out = (
        order_level.groupby("fulfilment", dropna=False)
        .agg(
            unique_orders=("Order ID", "nunique"),
            cancelled_orders=("is_cancelled", "sum"),
            recorded_amount=("recorded_amount", _sum),
        )
        .reset_index()
    )
    out["order_cancel_pct"] = out["cancelled_orders"] / out["unique_orders"] * 100
    record_rate = (
        amz.groupby("Fulfilment", dropna=False)["Status"]
        .apply(lambda s: s.eq("Cancelled").mean() * 100)
    )
    out["record_cancel_pct"] = out["fulfilment"].map(record_rate)
    return out


def quantity_by_status(amz: pd.DataFrame) -> pd.DataFrame:
    """Cell 181: zero-quantity records by status."""
    out = (
        amz.assign(qty_zero=amz["Qty"].eq(0))
        .groupby("Status", dropna=False)
        .agg(records=("Order ID", "size"), zero_qty=("qty_zero", "sum"))
        .reset_index()
        .sort_values("zero_qty", ascending=False)
    )
    out["zero_qty_pct"] = out["zero_qty"] / out["records"] * 100
    return out


# ===========================================================================
# International (de-duplicated)
# ===========================================================================
def intl_kpis(intl: pd.DataFrame) -> dict[str, float]:
    pcs = intl["pcs"].sum(min_count=1)
    gross = intl["gross_amt"].sum(min_count=1)
    return {
        "records": len(intl),
        "unique_customers": intl["customer"].nunique(),
        "unique_skus": intl["sku"].nunique(),
        "pieces": pcs,
        "gross_amount": gross,
        "gross_per_piece": gross / pcs if pcs else np.nan,
        "missing_sku": int(intl["sku"].isna().sum()),
        "date_min": intl["date"].min(),
        "date_max": intl["date"].max(),
    }


def intl_monthly(intl: pd.DataFrame) -> pd.DataFrame:
    """Cells 186-187: monthly performance, SKU completeness, gross per piece."""
    df = intl.assign(month=intl["date"].dt.to_period("M").dt.to_timestamp())
    out = (
        df.groupby("month")
        .agg(
            records=("customer", "size"),
            unique_customers=("customer", "nunique"),
            unique_skus=("sku", "nunique"),
            pieces=("pcs", "sum"),
            gross_amount=("gross_amt", "sum"),
            missing_sku_rows=("sku", lambda s: s.isna().sum()),
        )
        .reset_index()
        .sort_values("month")
    )
    out["missing_sku_pct"] = out["missing_sku_rows"] / out["records"] * 100
    out["gross_per_piece"] = out["gross_amount"] / out["pieces"].replace(0, np.nan)
    return out


def customer_summary(intl: pd.DataFrame) -> pd.DataFrame:
    """Cell 187: customer contribution with cumulative share."""
    out = (
        intl.groupby("customer", dropna=False)
        .agg(
            records=("customer", "size"),
            pieces=("pcs", "sum"),
            gross_amount=("gross_amt", "sum"),
            unique_skus=("sku", "nunique"),
        )
        .reset_index()
        .sort_values("gross_amount", ascending=False)
        .reset_index(drop=True)
    )
    total = out["gross_amount"].sum()
    out["share_pct"] = out["gross_amount"] / total * 100 if total else np.nan
    out["cumulative_share_pct"] = out["share_pct"].cumsum()
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def top_n_share(customers: pd.DataFrame, n: int) -> float:
    total = customers["gross_amount"].sum()
    return customers.head(n)["gross_amount"].sum() / total * 100 if total else np.nan


def sku_missingness_impact(intl: pd.DataFrame) -> pd.DataFrame:
    """Cell 187 (3): records with vs without SKU."""
    out = (
        intl.assign(sku_status=np.where(intl["sku"].isna(), "SKU missing", "SKU available"))
        .groupby("sku_status")
        .agg(
            records=("customer", "size"),
            customers=("customer", "nunique"),
            pieces=("pcs", "sum"),
            gross_amount=("gross_amt", "sum"),
        )
        .reset_index()
    )
    out["record_pct"] = out["records"] / out["records"].sum() * 100
    out["gross_pct"] = out["gross_amount"] / out["gross_amount"].sum() * 100
    return out


def dedup_comparison(full: pd.DataFrame, dedup: pd.DataFrame) -> pd.DataFrame:
    """Cell 164 audit: before vs after the candidate-key de-duplication."""
    rows = [
        ("Records", len(full), len(dedup)),
        ("Pieces", full["pcs"].sum(), dedup["pcs"].sum()),
        ("Gross amount", full["gross_amt"].sum(), dedup["gross_amt"].sum()),
    ]
    out = pd.DataFrame(rows, columns=["metric", "before", "after"])
    out["removed"] = out["before"] - out["after"]
    out["removed_pct"] = out["removed"] / out["before"] * 100
    return out


# ===========================================================================
# Product & inventory
# ===========================================================================
def product_kpis(prod: pd.DataFrame) -> dict[str, float]:
    stock = prod["stock"]
    return {
        "records": len(prod),
        "unique_skus": prod["sku"].nunique(),
        "missing_sku": int(prod["sku"].isna().sum()),
        "recorded_stock": stock.sum(min_count=1),
        "median_stock": stock.median(),
        "mean_stock": stock.mean(),
        "zero_stock_rows": int(stock.eq(0).sum()),
        "unique_designs": prod["design_no."].nunique(),
        "categories": prod["category"].nunique(),
    }


def inventory_by(prod: pd.DataFrame, col: str) -> pd.DataFrame:
    """Cell 193: stock by category / size / color (missing labelled)."""
    out = (
        prod.groupby(col, dropna=False)
        .agg(
            records=("stock", "size"),
            unique_skus=("sku", "nunique"),
            total_stock=("stock", "sum"),
            median_stock=("stock", "median"),
            zero_stock_rows=("stock", lambda s: s.eq(0).sum()),
        )
        .reset_index()
        .sort_values("total_stock", ascending=False)
    )
    total = out["total_stock"].sum()
    out["stock_share_pct"] = out["total_stock"] / total * 100 if total else np.nan
    out["record_pct"] = out["records"] / len(prod) * 100 if len(prod) else np.nan
    out[col] = out[col].astype("object").fillna(MISSING_LABEL)
    return out


def design_summary(prod: pd.DataFrame) -> pd.DataFrame:
    """Cell 329."""
    return (
        prod.groupby("design_no.")
        .agg(
            sku_count=("sku_standardized", "nunique"),
            total_stock=("stock", "sum"),
            avg_stock=("stock", "mean"),
        )
        .reset_index()
        .sort_values(["sku_count", "total_stock"], ascending=False)
    )


def duplicated_skus(prod: pd.DataFrame) -> pd.DataFrame:
    """Cell 330: SKUs recorded more than once (retained, not aggregated)."""
    mask = prod["sku_standardized"].notna() & prod["sku_standardized"].duplicated(keep=False)
    return prod.loc[mask].sort_values("sku_standardized")


def repeated_attribute_combinations(prod: pd.DataFrame) -> pd.DataFrame:
    """Cells 332-333."""
    key = ["design_no.", "category", "size", "color"]
    return prod[prod.duplicated(subset=key, keep=False)].sort_values(key)


def missing_sku_summary(prod: pd.DataFrame) -> pd.DataFrame:
    """Cell 334."""
    missing = prod[prod["sku_standardized"].isna()]
    out = (
        missing.groupby(["category", "size"], dropna=False)
        .agg(
            record_count=("stock", "size"),
            total_stock=("stock", "sum"),
            unique_designs=("design_no.", "nunique"),
        )
        .reset_index()
        .sort_values("record_count", ascending=False)
    )
    out["category"] = out["category"].astype("object").fillna(MISSING_LABEL)
    return out


# ===========================================================================
# Financial & operational
# ===========================================================================
def expense_reconciliation(expense: pd.DataFrame) -> dict[str, float]:
    """Cell 340 (section 6.5.3): the final row is the recorded summary."""
    transactions = expense.iloc[:-1]
    return {
        "gross_expense": transactions.loc[transactions["expense_amount"] > 0, "expense_amount"].sum(),
        "pending_amount": transactions.loc[
            transactions["expense_item"] == "Pending Amount", "expense_amount"
        ].sum(),
        "net_expense": transactions["expense_amount"].sum(),
        "recorded_summary": expense.iloc[-1]["expense_amount"],
    }


def expense_items(expense: pd.DataFrame) -> pd.DataFrame:
    """Positive expense items (summary row and pending adjustment excluded)."""
    tx = expense.iloc[:-1]
    out = tx[tx["expense_amount"] > 0].sort_values("expense_amount", ascending=False).copy()
    out["share_pct"] = out["expense_amount"] / out["expense_amount"].sum() * 100
    return out


# ===========================================================================
# Pricing reference (cells 189-191) — descriptive only
# ===========================================================================
def price_quality(df: pd.DataFrame, price_cols: list[str], name: str) -> pd.DataFrame:
    """Cell 189: valid / missing / zero / negative per price column."""
    rows = []
    for col in price_cols:
        v = df[col]
        rows.append({
            "dataset": name, "price_column": col,
            "valid": int(v.notna().sum()), "missing": int(v.isna().sum()),
            "zero": int(v.eq(0).sum()), "negative": int(v.lt(0).sum()),
            "min": v.min(), "median": v.median(), "max": v.max(),
        })
    return pd.DataFrame(rows)


def marketplace_consistency(may: pd.DataFrame, pnl: pd.DataFrame) -> pd.DataFrame:
    """Cell 190: are May-2022 and PnL-2021 prices identical per SKU?"""
    cols = MARKETPLACE_COLS + ["mrp_old", "final_mrp_old"]
    merged = may[["sku"] + cols].merge(pnl[["sku"] + cols], on="sku", how="outer",
                                       suffixes=("_may", "_pnl"))
    rows = []
    for col in cols:
        valid = merged[f"{col}_may"].notna() & merged[f"{col}_pnl"].notna()
        equal = merged.loc[valid, f"{col}_may"].eq(merged.loc[valid, f"{col}_pnl"])
        rows.append({
            "price_column": col, "comparable_skus": int(valid.sum()),
            "identical": int(equal.sum()), "different": int((~equal).sum()),
            "match_pct": equal.mean() * 100 if valid.sum() else np.nan,
        })
    return pd.DataFrame(rows)


def tp_relationship(may: pd.DataFrame, pnl: pd.DataFrame) -> dict[str, object]:
    """Cell 191: tp == tp_1, and tp_2 / tp_1 ~ 0.81."""
    tp = may[["sku", "tp"]].merge(pnl[["sku", "tp_1", "tp_2"]], on="sku", how="inner")
    valid_tp = tp[["tp", "tp_1"]].dropna()
    ratio = tp.dropna(subset=["tp_1", "tp_2"]).copy()
    ratio["tp_ratio"] = ratio["tp_2"] / ratio["tp_1"]
    exact = (ratio["tp_ratio"] - 0.81).abs() < 1e-6
    category = (
        pnl.assign(tp_ratio=pnl["tp_2"] / pnl["tp_1"])
        .groupby("category", dropna=False)
        .agg(sku_count=("sku", "nunique"), median_tp_1=("tp_1", "median"),
             median_tp_2=("tp_2", "median"), median_ratio=("tp_ratio", "median"))
        .reset_index()
        .sort_values("sku_count", ascending=False)
    )
    return {
        "tp_comparable": len(valid_tp),
        "tp_identical": int(valid_tp["tp"].eq(valid_tp["tp_1"]).sum()),
        "ratio_comparable": len(ratio),
        "ratio_exact_81": int(exact.sum()),
        "ratio_exact_81_pct": exact.mean() * 100,
        "ratio_values": ratio[["sku", "tp_1", "tp_2", "tp_ratio"]],
        "by_category": category,
    }

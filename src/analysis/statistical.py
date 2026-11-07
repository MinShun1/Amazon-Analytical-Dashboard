"""Hypothesis tests exactly as run in the notebook (cells 223-225, 270-294).

Every function returns plain Python/pandas objects so results can be saved
as artifacts and rendered without re-running the tests in the app.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

from src.config import ALPHA, INTL_SIZE_GROUPS


def cramers_v(contingency: pd.DataFrame, chi2: float) -> float:
    n = contingency.to_numpy().sum()
    denom = n * (min(contingency.shape) - 1)
    return float(np.sqrt(chi2 / denom)) if denom else np.nan


def epsilon_squared(h: float, n: int, k: int) -> float:
    return float((h - k + 1) / (n - k))


def _dunn_summary(dunn: pd.DataFrame) -> dict[str, Any]:
    upper = dunn.where(np.triu(np.ones(dunn.shape, dtype=bool), k=1)).stack()
    pairs = upper.reset_index()
    pairs.columns = ["group_a", "group_b", "p_adj"]
    pairs["significant"] = pairs["p_adj"] < ALPHA
    return {
        "matrix": dunn,
        "pairs": pairs,
        "significant_pairs": int(pairs["significant"].sum()),
        "total_pairs": len(pairs),
    }


# ---------------------------------------------------------------------------
# Test 4 — Fulfilment vs cancellation (cells 272-275)
# ---------------------------------------------------------------------------
def fulfilment_cancellation_chi2(amazon_ml: pd.DataFrame) -> dict[str, Any]:
    """Chi-square test of independence on the order-level Exp-1 dataset.
    Uses scipy's default Yates correction for 2x2 tables, as the notebook did."""
    table = pd.crosstab(amazon_ml["Fulfilment"], amazon_ml["cancelled"])
    chi2, p, dof, expected = chi2_contingency(table)
    rates = table.div(table.sum(axis=1), axis=0) * 100
    return {
        "contingency": table,
        "expected": pd.DataFrame(expected, index=table.index, columns=table.columns),
        "row_pct": rates,
        "chi2": float(chi2), "p_value": float(p), "dof": int(dof),
        "cramers_v": cramers_v(table, chi2),
        "min_expected": float(expected.min()),
        "n": int(table.to_numpy().sum()),
        "reject_h0": bool(p < ALPHA),
    }


# ---------------------------------------------------------------------------
# Test 5 — Amount across categories (cells 280-285)
# ---------------------------------------------------------------------------
def amount_test_sample(amazon: pd.DataFrame) -> pd.DataFrame:
    """Orders with a single category and complete Amount, aggregated to one
    observation per order; 'Dupatta' excluded for too few observations."""
    single = (
        amazon.groupby("Order ID")
        .agg(category_count=("Category", "nunique"),
             amount_complete=("Amount", lambda x: x.notna().all()))
        .query("category_count == 1 and amount_complete")
        .index
    )
    sample = (
        amazon[amazon["Order ID"].isin(single)]
        .groupby(["Order ID", "Category"], as_index=False)
        .agg(order_amount=("Amount", "sum"))
    )
    return sample[sample["Category"] != "Dupatta"]


def amount_by_category_kruskal(amazon: pd.DataFrame) -> dict[str, Any]:
    sample = amount_test_sample(amazon)
    groups = [g["order_amount"].values for _, g in sample.groupby("Category")]
    h, p = kruskal(*groups)
    n, k = len(sample), sample["Category"].nunique()
    summary = (
        sample.groupby("Category")["order_amount"]
        .agg(["count", "mean", "median", "std"])
        .sort_values("count", ascending=False)
    )
    dunn = sp.posthoc_dunn(sample, val_col="order_amount", group_col="Category", p_adjust="holm")
    return {
        "sample": sample, "summary": summary,
        "H": float(h), "p_value": float(p), "n": n, "k": k,
        "epsilon_squared": epsilon_squared(h, n, k),
        "reject_h0": bool(p < ALPHA),
        **{f"dunn_{key}": val for key, val in _dunn_summary(dunn).items()},
    }


# ---------------------------------------------------------------------------
# Test 6 — International purchasing behaviour (cells 288-294)
# ---------------------------------------------------------------------------
def frequency_rate_spearman(intl: pd.DataFrame) -> dict[str, Any]:
    customers = (
        intl.groupby("customer")
        .agg(purchase_frequency=("customer", "size"),
             avg_rate=("rate", "mean"), median_rate=("rate", "median"))
        .reset_index()
    )
    rho, p = spearmanr(customers["purchase_frequency"], customers["avg_rate"])
    return {"customers": customers, "rho": float(rho), "p_value": float(p),
            "n": len(customers), "reject_h0": bool(p < ALPHA)}


def rate_by_size_group_kruskal(intl: pd.DataFrame) -> dict[str, Any]:
    df = intl.copy()
    df["size_group"] = df["size"].map(INTL_SIZE_GROUPS)
    df = df.dropna(subset=["size_group", "rate"])
    groups = [g["rate"].values for _, g in df.groupby("size_group")]
    h, p = kruskal(*groups)
    n, k = len(df), df["size_group"].nunique()
    summary = (
        df.groupby("size_group")
        .agg(transaction_count=("rate", "count"), unique_styles=("style", "nunique"),
             avg_rate=("rate", "mean"), median_rate=("rate", "median"))
    )
    dunn = sp.posthoc_dunn(df, val_col="rate", group_col="size_group", p_adjust="holm")
    return {
        "sample": df[["customer", "style", "size", "size_group", "rate"]],
        "summary": summary,
        "H": float(h), "p_value": float(p), "n": n, "k": k,
        "epsilon_squared": epsilon_squared(h, n, k),
        "reject_h0": bool(p < ALPHA),
        **{f"dunn_{key}": val for key, val in _dunn_summary(dunn).items()},
    }


# ---------------------------------------------------------------------------
# Feature screening for Exp 1.1 (cells 223-225), April training data only
# ---------------------------------------------------------------------------
def feature_screening(train_features: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney (numeric) and chi-square (categorical, no Yates) against
    the cancellation target, Benjamini-Hochberg adjusted. Qty is excluded in
    the notebook because of target leakage."""
    df = train_features.copy()
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Ship_State"] = df["Ship_State"].fillna("Unknown")
    y = df["cancelled"]

    numeric = ["Item_Count", "Day_of_Week"]
    categorical = (
        ["Fulfilment", "Sales_Channel", "Ship_Service_Level", "Ship_State", "B2B"]
        + [c for c in df.columns if c.startswith("Category_")]
        + [c for c in df.columns if c.startswith("Size_")]
    )
    rows = []
    for f in numeric:
        a, b = df.loc[y == 1, f].dropna(), df.loc[y == 0, f].dropna()
        if a.nunique() <= 1 and b.nunique() <= 1:
            continue
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"Feature": f, "Test": "Mann-Whitney U", "Statistic": stat, "P_Value": p,
                     "Effect_Size": 2 * stat / (len(a) * len(b)) - 1, "Effect_Type": "Rank-biserial"})
    for f in categorical:
        table = pd.crosstab(df[f], y)
        if min(table.shape) < 2:
            continue
        chi2, p, _, _ = chi2_contingency(table, correction=False)
        rows.append({"Feature": f, "Test": "Chi-square", "Statistic": chi2, "P_Value": p,
                     "Effect_Size": cramers_v(table, chi2), "Effect_Type": "Cramer's V"})
    out = pd.DataFrame(rows)
    out["Adjusted_P_Value"] = multipletests(out["P_Value"], method="fdr_bh")[1]
    out["Significant"] = out["Adjusted_P_Value"] < ALPHA
    return out.sort_values("Adjusted_P_Value").reset_index(drop=True)


def ship_state_grouped_chi2(train_features: pd.DataFrame, rare_states: list[str],
                            state_mapping: dict[str, str]) -> dict[str, Any]:
    """Cell 225: chi-square after normalising and grouping rare states."""
    state = train_features["Ship_State"].fillna("Unknown").replace(state_mapping)
    grouped = state.where(~state.isin(rare_states), "Other")
    table = pd.crosstab(grouped, train_features["cancelled"])
    chi2, p, dof, expected = chi2_contingency(table, correction=False)
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof),
            "cramers_v": cramers_v(table, chi2), "min_expected": float(expected.min()),
            "expected_below_5": int((expected < 5).sum())}

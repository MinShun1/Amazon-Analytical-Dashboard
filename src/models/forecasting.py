"""International monthly sales forecasting experiment (cells 251-268).

Exploratory only: 11 complete monthly observations, one annual cycle.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing
from statsmodels.tsa.stattools import adfuller

INITIAL_TRAIN_SIZE = 6
LAST_COMPLETE_MONTH_EXCLUSIVE = "2022-05-01"   # May 2022 ends on the 11th


def monthly_sales(intl: pd.DataFrame) -> pd.DataFrame:
    """Cell 251 (+254 MoM growth)."""
    ts = intl.copy()
    ts["date"] = pd.to_datetime(ts["date"], errors="coerce")
    out = (
        ts.dropna(subset=["date"]).set_index("date").resample("MS")
        .agg(
            monthly_gross_amount=("gross_amt", "sum"),
            monthly_pieces=("pcs", "sum"),
            transaction_count=("customer", "count"),
            unique_customers=("customer", "nunique"),
            unique_skus=("sku_standardized", "nunique"),
        )
    )
    out["gross_amount_mom"] = out["monthly_gross_amount"].pct_change() * 100
    out["pieces_mom"] = out["monthly_pieces"].pct_change() * 100
    out["complete_month"] = out.index < LAST_COMPLETE_MONTH_EXCLUSIVE
    return out


def _metrics(group: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "MAE": mean_absolute_error(group["actual"], group["predicted"]),
        "RMSE": np.sqrt(mean_squared_error(group["actual"], group["predicted"])),
        "MAPE (%)": np.mean(np.abs((group["actual"] - group["predicted"]) / group["actual"])) * 100,
    })


def run_forecasting(intl: pd.DataFrame) -> dict[str, Any]:
    monthly = monthly_sales(intl)
    y = monthly.loc[monthly["complete_month"], "monthly_gross_amount"].sort_index().astype(float)

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # statsmodels convergence chatter on tiny samples
        for i in range(INITIAL_TRAIN_SIZE, len(y)):               # cells 263 + 267
            train, actual, when = y.iloc[:i], y.iloc[i], y.index[i]
            preds = {
                "Naive": train.iloc[-1],
                "Moving Average (3M)": train.iloc[-3:].mean(),
                "Simple Exponential Smoothing": SimpleExpSmoothing(
                    train, initialization_method="estimated").fit(optimized=True).forecast(1).iloc[0],
                "Holt's Linear Trend": Holt(
                    train, initialization_method="estimated").fit(optimized=True).forecast(1).iloc[0],
            }
            for model, pred in preds.items():
                rows.append({"date": when, "model": model, "actual": actual, "predicted": pred,
                             "absolute_error": abs(actual - pred)})
        adf = adfuller(y, regression="c", autolag="AIC")          # cell 261

    evaluation = pd.DataFrame(rows)
    evaluation["error"] = evaluation["actual"] - evaluation["predicted"]
    evaluation["ape_pct"] = evaluation["absolute_error"] / evaluation["actual"] * 100
    summary = (
        evaluation.groupby("model")[["actual", "predicted"]]
        .apply(_metrics)
        .sort_values("MAE")
        .reset_index()
    )
    rolling = pd.DataFrame({
        "actual": y, "rolling_mean_3m": y.rolling(3).mean(), "rolling_std_3m": y.rolling(3).std(),
    })
    return {
        "monthly": monthly.reset_index(), "series": y.rename("monthly_gross_amount").reset_index(),
        "evaluation": evaluation, "summary": summary, "rolling": rolling.reset_index(),
        "adf": {"statistic": float(adf[0]), "p_value": float(adf[1]), "used_lags": int(adf[2]),
                "n_obs": int(adf[3]), "critical_values": {k: float(v) for k, v in adf[4].items()}},
        "n_months": len(y),
    }

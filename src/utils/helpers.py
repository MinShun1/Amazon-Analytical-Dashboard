"""Consistent number, currency, percentage and date formatting."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

RUPEE = "₹"


def _is_missing(value: Any) -> bool:
    try:
        return value is None or pd.isna(value)
    except (TypeError, ValueError):
        return False


def fmt_int(value: Any) -> str:
    return "—" if _is_missing(value) else f"{value:,.0f}"


def fmt_num(value: Any, decimals: int = 2) -> str:
    return "—" if _is_missing(value) else f"{value:,.{decimals}f}"


def fmt_compact(value: Any, decimals: int = 2, prefix: str = "") -> str:
    """78,592,678 -> 78.59M (Indian lakh/crore avoided for an international audience)."""
    if _is_missing(value):
        return "—"
    v = float(value)
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"{prefix}{v / div:,.{decimals}f}{suffix}"
    return f"{prefix}{v:,.0f}"


def fmt_inr(value: Any, compact: bool = True) -> str:
    if _is_missing(value):
        return "—"
    return fmt_compact(value, prefix=RUPEE) if compact else f"{RUPEE}{value:,.2f}"


def fmt_pct(value: Any, decimals: int = 2) -> str:
    return "—" if _is_missing(value) else f"{value:,.{decimals}f}%"


def fmt_p(value: Any) -> str:
    """p-values: scientific below 0.001, never print a bare 0."""
    if _is_missing(value):
        return "—"
    if value == 0:
        return "< 1e-300"
    return f"{value:.2e}" if value < 0.001 else f"{value:.4f}"


def fmt_date(value: Any, fmt: str = "%d %b %Y") -> str:
    return "—" if _is_missing(value) else pd.Timestamp(value).strftime(fmt)


def fmt_month(value: Any) -> str:
    return fmt_date(value, "%b %Y")


def effect_label(value: float, kind: str) -> str:
    """Conventional magnitude labels (Cohen-style rules of thumb)."""
    if _is_missing(value):
        return "—"
    v = abs(value)
    thresholds = {
        "cramers_v": (0.1, 0.3, 0.5),
        "epsilon_squared": (0.01, 0.08, 0.26),
        "rho": (0.1, 0.3, 0.5),
    }[kind]
    if v < thresholds[0]:
        return "negligible"
    if v < thresholds[1]:
        return "small"
    if v < thresholds[2]:
        return "medium"
    return "large"


def safe_ratio(num: float, den: float) -> float:
    return num / den if den and not math.isnan(den) else float("nan")

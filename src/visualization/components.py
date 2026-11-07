"""Reusable Streamlit building blocks: headers, KPI cards, callouts,
method cards for hypothesis tests, filters and guards."""
from __future__ import annotations

from datetime import date
from typing import Any, Iterable, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import app_cache
from src.utils.helpers import fmt_p


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def page_header(title: str, subtitle: str, source: str | None = None) -> None:
    st.title(title)
    st.markdown(f"<p style='font-size:1.05rem;opacity:.8;margin-top:-.6rem'>{subtitle}</p>",
                unsafe_allow_html=True)
    if source:
        st.caption(source)


def page_link(path: str, label: str, container=None) -> None:
    """``st.page_link`` that degrades gracefully when the page is run outside
    the app's navigation (e.g. in headless tests)."""
    from streamlit.errors import StreamlitPageNotFoundError
    try:
        (container or st).page_link(path, label=label, icon=":material/arrow_forward:")
    except StreamlitPageNotFoundError:
        pass


def section(title: str, caption: str | None = None, divider: bool = True) -> None:
    if divider:
        st.divider()
    st.header(title)
    if caption:
        st.caption(caption)


def kpi_row(items: Sequence[dict[str, Any]], columns: int | None = None) -> None:
    """items: dicts with label, value and optional help / delta."""
    per_row = columns or min(len(items), 4)
    for start in range(0, len(items), per_row):
        row = items[start:start + per_row]
        cols = st.columns(per_row)
        for col, item in zip(cols, row):
            with col:
                st.metric(item["label"], item["value"], delta=item.get("delta"),
                          help=item.get("help"), border=True,
                          delta_color=item.get("delta_color", "normal"))


def chart(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def table(df: pd.DataFrame, column_config: dict | None = None, height: int | None = None,
          hide_index: bool = True) -> None:
    kwargs = {"height": height} if height else {}
    st.dataframe(df, width="stretch", hide_index=hide_index,
                 column_config=column_config, **kwargs)


# ---------------------------------------------------------------------------
# Callouts
# ---------------------------------------------------------------------------
def note(text: str, kind: str = "info") -> None:
    icons = {"info": ":material/info:", "warning": ":material/warning:",
             "limitation": ":material/report:", "success": ":material/check_circle:"}
    fn = {"info": st.info, "warning": st.warning, "limitation": st.warning,
          "success": st.success}[kind]
    fn(text, icon=icons[kind])


def insight_card(title: str, body: str, badges: Sequence[tuple[str, str]] = (),
                 footer: str | None = None) -> None:
    """Bordered card: title, coloured badges (text, colour), body and footer."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if badges:
            st.markdown(" ".join(f":{color}-badge[{text}]" for text, color in badges))
        st.markdown(body)
        if footer:
            st.caption(footer)


def card_grid(cards: Sequence[dict[str, Any]], columns: int = 2) -> None:
    for start in range(0, len(cards), columns):
        cols = st.columns(columns)
        for col, card in zip(cols, cards[start:start + columns]):
            with col:
                insight_card(**card)


def fixed_results_badge() -> None:
    st.caption(":material/lock: Fixed results from the notebook's analysis — not affected by "
               "the sidebar filters.")


def method_card(title: str, question: str, h0: str, h1: str, method: str,
                results: Sequence[tuple[str, str]], decision: str, interpretation: str,
                caveats: Iterable[str] = ()) -> None:
    with st.container(border=True):
        st.subheader(title)
        st.markdown(f"**Question.** {question}")
        c1, c2 = st.columns(2)
        c1.markdown(f"**H₀:** {h0}")
        c2.markdown(f"**H₁:** {h1}")
        st.markdown(f"**Method.** {method}")
        cols = st.columns(len(results))
        for col, (label, value) in zip(cols, results):
            col.metric(label, value)
        st.markdown(f"**Decision (α = 0.05).** {decision}")
        st.markdown(f"**Interpretation.** {interpretation}")
        for c in caveats:
            st.caption(f"• {c}")


def p_text(p: float) -> str:
    return f"p {'<' if p < 0.001 else '='} {('0.001' if p < 0.001 else fmt_p(p))}"


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def require_data() -> None:
    problem = app_cache.data_problem()
    if problem:
        st.error(f"**Data not available.** {problem}", icon=":material/error:")
        st.markdown("See *Getting the data* in the README, then run "
                    "`python scripts/build_artifacts.py`.")
        st.stop()


def require_results(bundle: str, what: str) -> tuple[dict[str, pd.DataFrame], dict[str, Any]] | None:
    res = app_cache.results(bundle)
    if res is None:
        st.warning(f"**{what} not built yet.** Run `python scripts/build_artifacts.py` from the "
                   "project root, then refresh this page.", icon=":material/build:")
    return res


def has_rows(df: pd.DataFrame, what: str = "the selected filters") -> bool:
    if df.empty:
        st.warning(f"No records match {what}. Widen the selection in the sidebar.",
                   icon=":material/filter_alt_off:")
        return False
    return True


# ---------------------------------------------------------------------------
# Filters (empty multiselect means "all")
# ---------------------------------------------------------------------------
def multiselect_filter(label: str, options: Iterable[Any], key: str,
                       help: str | None = None) -> list[Any]:
    opts = sorted({str(o) for o in options if pd.notna(o)})
    return st.sidebar.multiselect(label, opts, key=key, placeholder="All",
                                  help=help or "Leave empty to include all values.")


def apply_multiselect(df: pd.DataFrame, col: str, selected: list[Any]) -> pd.DataFrame:
    return df if not selected else df[df[col].astype(str).isin(selected)]


def date_range_filter(label: str, series: pd.Series, key: str) -> tuple[date, date]:
    valid = series.dropna()
    lo, hi = valid.min().date(), valid.max().date()
    picked = st.sidebar.date_input(label, value=(lo, hi), min_value=lo, max_value=hi, key=key)
    if isinstance(picked, (list, tuple)):
        if len(picked) == 2:
            start, end = picked
        elif len(picked) == 1:
            start = end = picked[0]
        else:
            start, end = lo, hi
    else:
        start = end = picked
    if start > end:
        st.sidebar.warning("Start date is after end date; showing the full range.")
        start, end = lo, hi
    return start, end


def filter_summary(n_shown: int, n_total: int, unit: str = "records") -> None:
    if n_shown == n_total:
        st.caption(f"Showing all {n_total:,} {unit}.")
    else:
        st.caption(f"Showing {n_shown:,} of {n_total:,} {unit} "
                   f"({n_shown / n_total * 100:.1f}%) after filters.")

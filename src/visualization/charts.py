"""Plotly chart builders with one shared look.

Every builder returns a ``go.Figure``; pages decide where to render it.
Colours: one accent for single-series charts, a fixed categorical palette
for groups, and a muted grey for context or "missing" categories.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ACCENT = "#2563EB"
ACCENT_SOFT = "#93C5FD"
POSITIVE = "#059669"
NEGATIVE = "#DC2626"
MUTED = "#94A3B8"
PALETTE = ["#2563EB", "#F59E0B", "#059669", "#DC2626", "#7C3AED",
           "#0891B2", "#DB2777", "#65A30D", "#EA580C", "#475569"]
SEQUENTIAL = "Blues"
DIVERGING = "RdBu_r"
FONT = "Inter, Segoe UI, system-ui, sans-serif"


def style(fig: go.Figure, title: str | None = None, height: int = 380,
          legend: bool = True, x_title: str | None = None,
          y_title: str | None = None) -> go.Figure:
    top = (48 if title else 16) + (28 if legend and title else 0)
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", y=0.98, yanchor="top",
                   font=dict(size=15)) if title else None,
        height=height,
        margin=dict(l=8, r=8, t=top, b=8),
        font=dict(family=FONT, size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_family=FONT),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        colorway=PALETTE,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, title=x_title, automargin=True)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.25)", zeroline=False, title=y_title,
                     automargin=True)
    return fig


# ---------------------------------------------------------------------------
# Basic charts
# ---------------------------------------------------------------------------
def line(df: pd.DataFrame, x: str, y: str | list[str], title: str, y_title: str,
         hover_format: str = ",.0f", color: str | None = None, markers: bool = True,
         x_title: str | None = None, height: int = 360) -> go.Figure:
    fig = px.line(df, x=x, y=y, color=color, markers=markers,
                  color_discrete_sequence=PALETTE)
    fig.update_traces(hovertemplate=f"%{{x}}<br>%{{y:{hover_format}}}<extra>%{{fullData.name}}</extra>")
    if isinstance(y, str) and color is None:
        fig.update_traces(line_color=ACCENT)
    return style(fig, title, height, legend=not isinstance(y, str) or color is not None,
                 x_title=x_title, y_title=y_title)


def bar(df: pd.DataFrame, x: str, y: str, title: str, *, horizontal: bool = False,
        value_format: str = ",.0f", color: str | None = None, text: bool = True,
        x_title: str | None = None, y_title: str | None = None, height: int = 380,
        highlight_missing: bool = True, color_map: dict | None = None,
        custom_hover: str | None = None) -> go.Figure:
    data = df.copy()
    if horizontal:
        fig = px.bar(data, x=y, y=x, orientation="h", color=color, barmode="group",
                     color_discrete_sequence=PALETTE, color_discrete_map=color_map)
        fig.update_yaxes(categoryorder="total ascending")
    else:
        fig = px.bar(data, x=x, y=y, color=color, barmode="group",
                     color_discrete_sequence=PALETTE, color_discrete_map=color_map)
    peak = pd.to_numeric(data[y], errors="coerce").max()
    if text and pd.notna(peak) and peak > 0:
        # Head-room so outside labels are not clipped.
        (fig.update_xaxes if horizontal else fig.update_yaxes)(range=[0, peak * 1.15])
    if color is None:
        cats = data[x].astype(str)
        colors = [MUTED if highlight_missing and c in ("Missing", "Unknown", "<NA>", "nan")
                  else ACCENT for c in cats]
        fig.update_traces(marker_color=colors)
    if text:
        fig.update_traces(texttemplate=f"%{{{'x' if horizontal else 'y'}:{value_format}}}",
                          textposition="outside", cliponaxis=False)
    axis_val = "x" if horizontal else "y"
    axis_cat = "y" if horizontal else "x"
    fig.update_traces(hovertemplate=custom_hover or
                      f"%{{{axis_cat}}}<br>%{{{axis_val}:{value_format}}}<extra></extra>")
    return style(fig, title, height, legend=color is not None,
                 x_title=x_title, y_title=y_title)


def histogram(values: pd.Series, title: str, x_title: str, nbins: int = 40,
              log_y: bool = False, height: int = 340) -> go.Figure:
    fig = px.histogram(values.dropna(), nbins=nbins, log_y=log_y)
    fig.update_traces(marker_color=ACCENT, hovertemplate="%{x}<br>Count: %{y:,}<extra></extra>")
    return style(fig, title, height, legend=False, x_title=x_title,
                 y_title="Count (log)" if log_y else "Count")


def box(df: pd.DataFrame, x: str, y: str, title: str, y_title: str,
        order: Sequence[str] | None = None, x_title: str | None = None,
        points: bool | str = False, height: int = 400, log_y: bool = False) -> go.Figure:
    fig = px.box(df, x=x, y=y, points=points, category_orders={x: list(order)} if order else None,
                 log_y=log_y)
    fig.update_traces(marker_color=ACCENT, line_color=ACCENT, fillcolor="rgba(37,99,235,0.15)")
    return style(fig, title, height, legend=False, x_title=x_title, y_title=y_title)


def heatmap(matrix: pd.DataFrame, title: str, fmt: str = ".2f", colorscale: str = SEQUENTIAL,
            zmid: float | None = None, x_title: str | None = None, y_title: str | None = None,
            height: int = 420, colorbar_title: str | None = None, zmin=None, zmax=None) -> go.Figure:
    z = matrix.to_numpy(dtype=float)
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(c) for c in matrix.columns], y=[str(i) for i in matrix.index],
        colorscale=colorscale, zmid=zmid, zmin=zmin, zmax=zmax,
        text=np.vectorize(lambda v: "" if np.isnan(v) else format(v, fmt))(z),
        texttemplate="%{text}", hovertemplate="%{y} × %{x}<br>%{z:" + fmt + "}<extra></extra>",
        colorbar=dict(title=colorbar_title, thickness=12),
    ))
    fig.update_yaxes(autorange="reversed")
    return style(fig, title, height, legend=False, x_title=x_title, y_title=y_title)


def scatter(df: pd.DataFrame, x: str, y: str, title: str, *, color: str | None = None,
            hover: Iterable[str] | None = None, log_x: bool = False, log_y: bool = False,
            x_title: str | None = None, y_title: str | None = None, height: int = 420,
            size: str | None = None, color_map: dict | None = None,
            category_orders: dict | None = None, opacity: float = 0.75) -> go.Figure:
    fig = px.scatter(df, x=x, y=y, color=color, hover_name=None, hover_data=list(hover or []),
                     log_x=log_x, log_y=log_y, size=size, color_discrete_sequence=PALETTE,
                     color_discrete_map=color_map, category_orders=category_orders,
                     opacity=opacity)
    if color is None:
        fig.update_traces(marker_color=ACCENT)
    return style(fig, title, height, legend=color is not None, x_title=x_title, y_title=y_title)


# ---------------------------------------------------------------------------
# Analytical charts
# ---------------------------------------------------------------------------
def confusion_matrix(cm: list[list[int]], title: str,
                     labels: Sequence[str] = ("Not Cancelled", "Cancelled"),
                     height: int = 360) -> go.Figure:
    z = np.array(cm, dtype=float)
    row_pct = z / z.sum(axis=1, keepdims=True) * 100
    text = [[f"{int(z[i, j]):,}<br>({row_pct[i, j]:.1f}% of actual)" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=row_pct, x=[f"Predicted: {l}" for l in labels], y=[f"Actual: {l}" for l in labels],
        colorscale=SEQUENTIAL, zmin=0, zmax=100, text=text, texttemplate="%{text}",
        hovertemplate="%{y}<br>%{x}<br>%{text}<extra></extra>", showscale=False,
    ))
    fig.update_yaxes(autorange="reversed")
    return style(fig, title, height, legend=False)


def probability_distribution(df: pd.DataFrame, prob_col: str, target_col: str,
                             threshold: float, title: str, height: int = 340) -> go.Figure:
    d = df[[prob_col, target_col]].copy()
    d["Actual"] = np.where(d[target_col] == 1, "Cancelled", "Not cancelled")
    fig = px.histogram(d, x=prob_col, color="Actual", nbins=50, barmode="overlay",
                       histnorm="percent", opacity=0.6,
                       color_discrete_map={"Cancelled": NEGATIVE, "Not cancelled": ACCENT})
    fig.add_vline(x=threshold, line_dash="dash", line_color="#334155",
                  annotation_text=f"threshold {threshold:.2f}", annotation_position="top")
    fig.update_traces(hovertemplate="p=%{x:.2f}<br>%{y:.1f}% of class<extra></extra>")
    return style(fig, title, height, x_title="Predicted probability of cancellation",
                 y_title="% of orders in class")


def metric_comparison(metrics: pd.DataFrame, metric_cols: Sequence[str], title: str,
                      baseline: dict[str, float] | None = None, height: int = 420) -> go.Figure:
    long = metrics.melt(id_vars="Model", value_vars=list(metric_cols),
                        var_name="Metric", value_name="Value")
    fig = px.bar(long, x="Metric", y="Value", color="Model", barmode="group",
                 color_discrete_sequence=PALETTE)
    fig.update_traces(hovertemplate="%{fullData.name}<br>%{x}: %{y:.4f}<extra></extra>")
    for name, value in (baseline or {}).items():
        fig.add_hline(y=value, line_dash="dot", line_color=MUTED,
                      annotation_text=name, annotation_position="top left")
    fig.update_yaxes(range=[0, 1])
    return style(fig, title, height, y_title="Score")


def waterfall(labels: Sequence[str], values: Sequence[float], measures: Sequence[str],
              title: str, height: int = 360) -> go.Figure:
    running, text = 0.0, []
    for v, m in zip(values, measures):
        if m == "total":
            text.append(f"{running:,.0f}")
        else:
            running += v
            text.append(f"{v:,.0f}")
    fig = go.Figure(go.Waterfall(
        x=list(labels), y=list(values), measure=list(measures),
        text=text, textposition="outside",
        increasing=dict(marker_color=ACCENT), decreasing=dict(marker_color=NEGATIVE),
        totals=dict(marker_color=POSITIVE), connector=dict(line_color=MUTED),
        hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>",
    ))
    return style(fig, title, height, legend=False)


def pareto(customers: pd.DataFrame, title: str, top_n: int = 25, height: int = 420) -> go.Figure:
    top = customers.head(top_n)
    fig = go.Figure()
    fig.add_bar(x=top["customer"], y=top["gross_amount"], name="Gross amount",
                marker_color=ACCENT, hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>")
    fig.add_scatter(x=top["customer"], y=top["cumulative_share_pct"], name="Cumulative share",
                    yaxis="y2", mode="lines+markers", line_color="#F59E0B",
                    hovertemplate="%{x}<br>Cumulative: %{y:.1f}%<extra></extra>")
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 100], dtick=25,
                                  ticksuffix="%", showgrid=False, title="Cumulative share"))
    fig.update_xaxes(tickangle=-40)
    return style(fig, title, height, y_title="Gross amount")

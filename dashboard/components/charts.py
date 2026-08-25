"""Plotly figures for the dashboard.

Pure functions: data in, figure out. No Streamlit calls, no HTTP -- so a chart
can be built and inspected in a notebook or a test without a browser.

One convention runs through all of them: the uncertainty band is drawn *before*
the line it belongs to, so the line sits on top. Plotly draws in trace order,
and a band drawn last hides the forecast it is supposed to qualify.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go

from dashboard.components.ui import COLOURS


def _layout(figure: go.Figure, *, title: str, y_title: str, height: int = 380) -> go.Figure:
    figure.update_layout(
        title=title,
        yaxis_title=y_title,
        height=height,
        margin=dict(l=10, r=10, t=48, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="rgba(148,163,184,0.20)")
    return figure


def forecast_chart(points: List[Dict[str, Any]], *, title: str = "Demand forecast") -> go.Figure:
    """Forecast with its 80% band and the underlying trend.

    The band is the point of this chart. A forecast line alone invites a reader
    to treat 0.82 as a fact; the band shows it is 0.74 to 0.90 and that pricing
    decisions should respect the difference.
    """
    frame = pd.DataFrame(points)
    figure = go.Figure()

    if frame.empty:
        return _layout(figure, title=title, y_title="Demand")

    figure.add_trace(
        go.Scatter(
            x=frame["date"], y=frame["upper"], mode="lines",
            line=dict(width=0), hoverinfo="skip", showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["date"], y=frame["lower"], mode="lines", fill="tonexty",
            fillcolor=COLOURS["band"], line=dict(width=0),
            name="80% interval", hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["date"], y=frame["forecast"], mode="lines+markers",
            name="Forecast", line=dict(color=COLOURS["forecast"], width=2.5),
            marker=dict(size=5),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["date"], y=frame["trend"], mode="lines", name="Trend",
            line=dict(color=COLOURS["muted"], width=1.5, dash="dot"),
        )
    )

    figure = _layout(figure, title=title, y_title="Demand (share of inventory)")
    figure.update_yaxes(tickformat=".0%", rangemode="tozero")
    return figure


def price_vs_market_chart(
    summaries: List[Dict[str, Any]], *, our_price: Optional[float] = None
) -> go.Figure:
    """The competitive band per night, with our rate laid over it."""
    frame = pd.DataFrame(summaries)
    figure = go.Figure()

    if frame.empty:
        return _layout(figure, title="Competitor rates", y_title="Rate")

    figure.add_trace(
        go.Scatter(
            x=frame["check_in_date"], y=frame["competitor_max_rate"], mode="lines",
            line=dict(width=0), hoverinfo="skip", showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["check_in_date"], y=frame["competitor_min_rate"], mode="lines",
            fill="tonexty", fillcolor="rgba(148,163,184,0.25)", line=dict(width=0),
            name="Competitor range",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["check_in_date"], y=frame["competitor_rate"], mode="lines+markers",
            name="Market average", line=dict(color=COLOURS["muted"], width=2),
        )
    )

    if our_price is not None:
        figure.add_hline(
            y=our_price,
            line=dict(color=COLOURS["primary"], width=2, dash="dash"),
            annotation_text=f"Our rate {our_price:,.0f}",
            annotation_position="top left",
        )

    return _layout(figure, title="Where we sit against the market", y_title="Rate (INR)")


def price_history_chart(items: List[Dict[str, Any]]) -> go.Figure:
    """Recommended prices over time, with guardrailed decisions marked.

    Raw and final are both drawn: the gap between them is the guardrails doing
    their job, and it is the most useful thing on the page. A wide, persistent
    gap means the model wants prices the business will not allow -- which is a
    retuning signal, not a success.
    """
    frame = pd.DataFrame(items)
    figure = go.Figure()

    if frame.empty:
        return _layout(figure, title="Pricing history", y_title="Rate")

    frame = frame.sort_values("created_at")
    frame["clamped"] = frame["guardrails_applied"].apply(bool)

    figure.add_trace(
        go.Scatter(
            x=frame["created_at"], y=frame["raw_recommended_price"], mode="lines",
            name="Raw (before guardrails)",
            line=dict(color=COLOURS["muted"], width=1.5, dash="dot"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["created_at"], y=frame["final_recommended_price"], mode="lines+markers",
            name="Final (served)", line=dict(color=COLOURS["primary"], width=2.5),
            marker=dict(size=6),
        )
    )

    clamped = frame[frame["clamped"]]
    if not clamped.empty:
        figure.add_trace(
            go.Scatter(
                x=clamped["created_at"], y=clamped["final_recommended_price"],
                mode="markers", name="Guardrail applied",
                marker=dict(color=COLOURS["danger"], size=11, symbol="x"),
            )
        )

    return _layout(figure, title="Recommended prices over time", y_title="Rate (INR)")


def adjustment_waterfall(adjustments: List[Dict[str, Any]], base_price: float) -> go.Figure:
    """The price build-up, one bar per adjustment.

    A waterfall because that is what the arithmetic *is*: start at base, add
    five signed percentages, arrive at the raw price. Any other chart type would
    obscure the one property that makes the engine defensible.
    """
    figure = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute"] + ["relative"] * len(adjustments) + ["total"],
            x=["Base"] + [a["name"].title() for a in adjustments] + ["Raw price"],
            y=[base_price] + [base_price * a["value"] for a in adjustments] + [0],
            text=[f"{base_price:,.0f}"]
            + [f"{a['percent']:+.1f}%" for a in adjustments]
            + [""],
            textposition="outside",
            connector=dict(line=dict(color="rgba(148,163,184,0.5)")),
            increasing=dict(marker=dict(color=COLOURS["success"])),
            decreasing=dict(marker=dict(color=COLOURS["danger"])),
            totals=dict(marker=dict(color=COLOURS["primary"])),
        )
    )
    return _layout(figure, title="How the price was built", y_title="Rate (INR)", height=420)


def feature_importance_chart(rows: List[Dict[str, Any]], *, top_n: int = 15) -> go.Figure:
    """Permutation importance, largest first.

    Sorts defensively rather than trusting the caller. The producing function
    already returns sorted rows, but a chart that silently mis-ranks because
    someone passed a raw list is a bug that looks like a finding.
    """
    figure = go.Figure()
    if not rows:
        return _layout(figure, title="Feature importance", y_title="")

    # Reversed after sorting: Plotly draws a horizontal bar chart bottom-up, so
    # ascending order puts the largest bar at the top.
    frame = (
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False)
        .head(top_n)
        .iloc[::-1]
    )

    figure.add_trace(
        go.Bar(
            x=frame["importance"], y=frame["feature"], orientation="h",
            marker=dict(color=COLOURS["primary"]),
            error_x=dict(array=frame["std"]) if "std" in frame else None,
        )
    )
    figure = _layout(
        figure,
        title="Permutation importance (holdout)",
        y_title="",
        height=max(340, 26 * len(frame)),
    )
    figure.update_xaxes(title="Increase in MAE when the feature is scrambled")
    return figure


def metric_by_horizon_chart(rows: List[Dict[str, Any]]) -> go.Figure:
    """Model error against the baseline, by days to check-in.

    The shape is the story: error should be lowest near check-in, where the
    booking curve has already resolved most of the answer, and highest far out.
    A flat line means the model is not using the curve.
    """
    figure = go.Figure()
    if not rows:
        # Guard before sort_values: sorting an empty frame by a column it does
        # not have raises KeyError, and every chart here is reachable before any
        # data exists.
        return _layout(figure, title="Accuracy by lead time", y_title="MAE")

    frame = pd.DataFrame(rows).sort_values("days_to_checkin")

    figure.add_trace(
        go.Scatter(
            x=frame["days_to_checkin"], y=frame["baseline_mae"], name="Baseline (mean)",
            line=dict(color=COLOURS["muted"], width=2, dash="dot"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["days_to_checkin"], y=frame["mae"], name="Model",
            line=dict(color=COLOURS["primary"], width=2.5), mode="lines+markers",
        )
    )
    figure = _layout(figure, title="Accuracy by days to check-in", y_title="MAE (lower is better)")
    figure.update_xaxes(title="Days to check-in")
    return figure


def distribution_chart(
    values: List[float], *, title: str, x_title: str, reference: Optional[float] = None
) -> go.Figure:
    """A histogram, with an optional reference line.

    Used for prediction and price distributions on the monitoring page: the
    shape moving is the earliest visible sign of drift, well before any accuracy
    metric reacts.
    """
    figure = go.Figure()
    if not values:
        return _layout(figure, title=title, y_title="Count")

    figure.add_trace(
        go.Histogram(x=values, marker=dict(color=COLOURS["primary"]), nbinsx=30)
    )
    if reference is not None:
        figure.add_vline(
            x=reference,
            line=dict(color=COLOURS["danger"], width=2, dash="dash"),
            annotation_text=f"{reference:,.2f}",
        )
    figure = _layout(figure, title=title, y_title="Count")
    figure.update_xaxes(title=x_title)
    return figure


__all__ = [
    "adjustment_waterfall",
    "distribution_chart",
    "feature_importance_chart",
    "forecast_chart",
    "metric_by_horizon_chart",
    "price_history_chart",
    "price_vs_market_chart",
]

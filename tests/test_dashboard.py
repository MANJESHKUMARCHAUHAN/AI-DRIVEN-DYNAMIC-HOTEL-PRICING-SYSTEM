"""Tests for the dashboard.

Three layers, and only the third needs anything running:

* **Charts** are pure functions -- data in, figure out -- so they are tested
  directly. The assertions are about *structure*, because that is what a
  regression would break: an uncertainty band drawn after the line it qualifies
  hides the forecast, and no visual test catches that.
* **The API client** is tested with the network stubbed out. Its job is to turn
  every failure mode into a renderable result rather than a traceback, which is
  exactly the behaviour a dashboard needs and the easiest to get wrong.
* **The pages** are executed with Streamlit's ``AppTest`` against a live API.
  Those are skipped when nothing is listening, so the suite stays green on a
  machine with no stack running.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pytest
import requests

from dashboard import api_client
from dashboard.components import charts

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Where the page tests expect to find a running API.
LIVE_API = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def _api_is_live() -> bool:
    try:
        return requests.get(f"{LIVE_API}/health", timeout=2).status_code == 200
    except requests.exceptions.RequestException:
        return False


requires_api = pytest.mark.skipif(
    not _api_is_live(),
    reason=f"no API listening at {LIVE_API}; start it with `uvicorn api.main:app`",
)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #


def forecast_points(n: int = 7) -> List[Dict[str, Any]]:
    start = date(2026, 9, 1)
    return [
        {
            "date": (start + timedelta(days=offset)).isoformat(),
            "forecast": 0.70 + offset * 0.01,
            "lower": 0.62 + offset * 0.01,
            "upper": 0.78 + offset * 0.01,
            "trend": 0.69,
        }
        for offset in range(n)
    ]


class TestForecastChart:
    def test_draws_the_band_before_the_line(self) -> None:
        """Plotly draws in trace order. A band added last covers the forecast
        it is supposed to qualify."""
        figure = charts.forecast_chart(forecast_points())
        names = [trace.name for trace in figure.data]
        assert names.index("80% interval") < names.index("Forecast")

    def test_has_forecast_band_and_trend(self) -> None:
        figure = charts.forecast_chart(forecast_points())
        assert {"80% interval", "Forecast", "Trend"} <= {
            t.name for t in figure.data if t.name
        }

    def test_survives_an_empty_series(self) -> None:
        """Every chart is reachable before any data exists."""
        assert charts.forecast_chart([]).data == ()

    def test_y_axis_is_a_percentage(self) -> None:
        figure = charts.forecast_chart(forecast_points())
        assert figure.layout.yaxis.tickformat == ".0%"


class TestAdjustmentWaterfall:
    def _adjustments(self) -> List[Dict[str, Any]]:
        return [
            {"name": "demand", "value": 0.12, "percent": 12.0},
            {"name": "occupancy", "value": 0.08, "percent": 8.0},
            {"name": "season", "value": -0.08, "percent": -8.0},
        ]

    def test_starts_at_base_and_ends_at_the_total(self) -> None:
        figure = charts.adjustment_waterfall(self._adjustments(), 5_000.0)
        measures = list(figure.data[0].measure)
        assert measures[0] == "absolute"
        assert measures[-1] == "total"
        assert measures.count("relative") == 3

    def test_each_adjustment_gets_a_bar(self) -> None:
        figure = charts.adjustment_waterfall(self._adjustments(), 5_000.0)
        assert len(figure.data[0].x) == 5  # base + 3 adjustments + total

    def test_bars_are_the_rupee_value_of_each_percentage(self) -> None:
        figure = charts.adjustment_waterfall(self._adjustments(), 5_000.0)
        assert figure.data[0].y[1] == pytest.approx(600.0)  # 12% of 5000
        assert figure.data[0].y[3] == pytest.approx(-400.0)  # -8% of 5000


class TestPriceHistoryChart:
    def _items(self) -> List[Dict[str, Any]]:
        return [
            {
                "created_at": f"2026-09-0{n}T10:00:00Z",
                "raw_recommended_price": 7000 + n * 100,
                "final_recommended_price": 6800 + n * 50,
                "guardrails_applied": ["MAX_DAILY_RISE"] if n % 2 else [],
            }
            for n in range(1, 6)
        ]

    def test_shows_raw_and_final(self) -> None:
        """The gap between them is the guardrails working, and it is the most
        useful thing on the chart."""
        names = {t.name for t in charts.price_history_chart(self._items()).data}
        assert "Raw (before guardrails)" in names
        assert "Final (served)" in names

    def test_marks_clamped_decisions(self) -> None:
        figure = charts.price_history_chart(self._items())
        marked = [t for t in figure.data if t.name == "Guardrail applied"]
        assert marked and len(marked[0].x) == 3

    def test_no_marker_trace_when_nothing_was_clamped(self) -> None:
        clean = [{**item, "guardrails_applied": []} for item in self._items()]
        names = {t.name for t in charts.price_history_chart(clean).data}
        assert "Guardrail applied" not in names


class TestOtherCharts:
    def test_competitor_band_is_drawn_under_the_average(self) -> None:
        summaries = [
            {
                "check_in_date": "2026-09-01",
                "competitor_rate": 6500,
                "competitor_min_rate": 6200,
                "competitor_max_rate": 6800,
            }
        ]
        figure = charts.price_vs_market_chart(summaries, our_price=6400)
        names = [t.name for t in figure.data]
        assert names.index("Competitor range") < names.index("Market average")

    def test_feature_importance_is_sorted_largest_first(self) -> None:
        rows = [
            {"feature": "a", "importance": 0.01, "std": 0.001},
            {"feature": "b", "importance": 0.03, "std": 0.001},
        ]
        figure = charts.feature_importance_chart(rows)
        # The frame is reversed so the largest bar sits at the top of a
        # horizontal chart, which means the last y value is the biggest.
        assert list(figure.data[0].y)[-1] == "b"

    def test_horizon_chart_plots_model_against_baseline(self) -> None:
        rows = [
            {"days_to_checkin": 0, "mae": 0.04, "baseline_mae": 0.14},
            {"days_to_checkin": 30, "mae": 0.08, "baseline_mae": 0.15},
        ]
        names = {t.name for t in charts.metric_by_horizon_chart(rows).data}
        assert names == {"Model", "Baseline (mean)"}

    def test_distribution_chart_marks_the_reference(self) -> None:
        figure = charts.distribution_chart(
            [0.5, 0.6, 0.7], title="t", x_title="x", reference=0.6
        )
        assert figure.layout.shapes  # the vline

    @pytest.mark.parametrize(
        "builder",
        [
            lambda: charts.price_vs_market_chart([]),
            lambda: charts.price_history_chart([]),
            lambda: charts.feature_importance_chart([]),
            lambda: charts.metric_by_horizon_chart([]),
            lambda: charts.distribution_chart([], title="t", x_title="x"),
        ],
    )
    def test_every_chart_survives_no_data(self, builder) -> None:
        assert builder() is not None


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #


class TestApiClient:
    def test_a_refused_connection_becomes_a_result(self, monkeypatch) -> None:
        """A dashboard that shows a traceback when the API restarts is worse
        than one that says which URL it tried."""

        def _refuse(*args, **kwargs):
            raise requests.exceptions.ConnectionError()

        monkeypatch.setattr(requests, "request", _refuse)
        result = api_client.health()

        assert result.ok is False
        assert api_client.API_BASE_URL in result.error

    def test_a_timeout_becomes_a_result(self, monkeypatch) -> None:
        def _timeout(*args, **kwargs):
            raise requests.exceptions.Timeout()

        monkeypatch.setattr(requests, "request", _timeout)
        assert api_client.models().ok is False

    def test_an_error_body_surfaces_the_api_detail(self, monkeypatch) -> None:
        class Response:
            status_code = 404

            @staticmethod
            def json():
                return {"error": "http_404", "detail": "No hotel with id 'H999'"}

        monkeypatch.setattr(requests, "request", lambda *a, **k: Response())
        result = api_client.get_hotel("H999")

        assert result.ok is False
        assert result.status_code == 404
        assert "H999" in result.error

    def test_a_non_json_body_is_reported_not_raised(self, monkeypatch) -> None:
        class Response:
            status_code = 200
            text = "<html>gateway</html>"

            @staticmethod
            def json():
                raise ValueError("not json")

        monkeypatch.setattr(requests, "request", lambda *a, **k: Response())
        assert api_client.list_hotels().ok is False

    def test_unwrap_returns_the_default_on_failure(self) -> None:
        assert api_client.ApiResult(ok=False, error="x").unwrap([]) == []

    def test_health_bypasses_the_api_prefix(self) -> None:
        """/health sits at the root so healthchecks find it regardless of the
        configured prefix."""
        assert api_client._url("/health").endswith("/health")
        assert api_client.API_PREFIX not in api_client._url("/health")

    def test_business_paths_carry_the_prefix(self) -> None:
        assert api_client._url("/hotels").endswith(f"{api_client.API_PREFIX}/hotels")

    def test_optional_pricing_fields_are_omitted_when_absent(self, monkeypatch) -> None:
        """Sending nulls would override the API's own lookups with nothing."""
        captured: Dict[str, Any] = {}

        class Response:
            status_code = 200

            @staticmethod
            def json():
                return {}

        def _capture(method, url, params=None, json=None, timeout=None, headers=None):
            captured.update(json or {})
            return Response()

        monkeypatch.setattr(requests, "request", _capture)
        api_client.predict_price(
            hotel_id="H001", room_type="deluxe", check_in_date=date(2026, 9, 15)
        )

        assert "current_price" not in captured
        assert "competitor_rate" not in captured
        assert captured["hotel_id"] == "H001"


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


PAGES = ["dashboard/app.py"] + sorted(
    str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    for path in (PROJECT_ROOT / "dashboard" / "pages").glob("*.py")
)


@requires_api
class TestPagesRender:
    """Executes each page against a live API.

    This is the closest thing the project has to an end-to-end test: a page that
    renders proves the API endpoint behind it works, that the response shape
    matches what the dashboard expects, and that the charts accept real data.
    """

    @pytest.mark.parametrize("page", PAGES)
    def test_page_runs_without_raising(self, page: str) -> None:
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PROJECT_ROOT / page), default_timeout=120)
        app.run()

        assert not app.exception, [e.value for e in app.exception]

    def test_the_overview_reports_the_estate(self) -> None:
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py"), default_timeout=120)
        app.run()

        labels = {metric.label for metric in app.metric}
        assert {"Hotels", "Rooms", "Cities"} <= labels


# --------------------------------------------------------------------------- #
# The optional AI Agent page
# --------------------------------------------------------------------------- #


class TestAgentPageDegrades:
    """The agent is optional, and "optional" is a claim that needs testing.

    No live API required: the interesting path is the one where the SDK or the
    key is absent, which is the state most people will first open this page in.
    A traceback there would make an optional add-on look like a broken build.
    """

    AGENT_PAGE = PROJECT_ROOT / "dashboard" / "pages" / "8_AI_Agent.py"

    @staticmethod
    def _force_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
        """Pin the page to the unavailable branch.

        Patched rather than left to chance so this asserts the same thing on a
        developer machine that happens to have the SDK and a key exported.
        """
        from ai_agent import tools

        monkeypatch.setattr(
            tools,
            "agent_status",
            lambda: {
                "available": False,
                "sdk_installed": False,
                "api_key_present": False,
                "api_base_url": LIVE_API,
                "tool_count": len(tools.RAW_TOOLS),
                "problems": ["The anthropic SDK is not installed."],
            },
        )

    def test_renders_install_instructions_instead_of_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from streamlit.testing.v1 import AppTest

        self._force_unavailable(monkeypatch)

        app = AppTest.from_file(str(self.AGENT_PAGE), default_timeout=60)
        app.run()

        assert not app.exception, [e.value for e in app.exception]

        rendered = " ".join(block.value for block in app.code)
        assert "[agent]" in rendered, (
            "the page must say what to install, not just that something is missing"
        )

    def test_states_that_it_cannot_price(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Where the boundary sits is a product claim, so it belongs on screen."""
        from streamlit.testing.v1 import AppTest

        self._force_unavailable(monkeypatch)

        app = AppTest.from_file(str(self.AGENT_PAGE), default_timeout=60)
        app.run()

        prose = " ".join(block.value for block in app.markdown).lower()
        assert "does not price" in prose or "cannot write" in prose

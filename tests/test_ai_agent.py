"""The agent's tool layer.

The property under test is the one the whole design rests on: **this package
cannot write.** Not "is instructed not to" -- cannot. If somebody later adds a
tool that submits a rate, these tests fail, which is the point.

The model itself is not tested here. Asserting on generated prose is a test that
fails for reasons unrelated to the code, and the SDK is an optional dependency
the suite must pass without.
"""

from __future__ import annotations

import json

import pytest

from ai_agent import tools


class TestAllowlist:
    """The structural guarantee, tested from both sides."""

    @pytest.mark.parametrize(
        "method,path",
        [
            ("GET", "/hotels"),
            ("GET", "/pricing/H001"),
            ("GET", "/competitors/H001"),
            ("GET", "/forecast/H001"),
            ("GET", "/ingestion/status"),
            ("POST", "/pricing/predict"),
        ],
    )
    def test_reads_and_simulations_are_allowed(self, method: str, path: str) -> None:
        assert tools._is_allowed(method, path)

    @pytest.mark.parametrize(
        "method,path",
        [
            ("POST", "/competitors/events"),   # would write a competitor rate
            ("POST", "/models/train"),         # would burn a training run
            ("POST", "/ingestion/run"),        # would scrape and persist
            ("DELETE", "/hotels/H001"),
            ("PUT", "/pricing/H001"),
            ("GET", "/admin"),
        ],
    )
    def test_everything_else_is_denied(self, method: str, path: str) -> None:
        assert not tools._is_allowed(method, path)

    def test_denied_call_raises_rather_than_returning_an_error(self) -> None:
        """A blocked call is a bug in this module, not a condition for the model.

        Tool failures are returned as strings so the model can reason about them.
        This one is different: it means somebody wired up a tool that should not
        exist, and it must stop the process rather than become a sentence in an
        answer.
        """
        with pytest.raises(tools.ToolCallDenied, match="allowlist"):
            tools._request("POST", "/competitors/events")

    def test_read_methods_do_not_cover_write_verbs_on_the_same_path(self) -> None:
        """``GET /pricing/`` being allowed must not imply ``POST /pricing/``."""
        assert tools._is_allowed("GET", "/pricing/H001")
        assert not tools._is_allowed("POST", "/pricing/H001")


class TestSimulateAlwaysPreviews:
    def test_persist_is_hardcoded_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The audit trail must be unreachable from here.

        ``persist`` is not a tool parameter, so the model has no way to set it.
        If it were exposed, the allowlist would be the only thing standing
        between an explanation and a written price.
        """
        captured = {}

        class _Response:
            status_code = 200

            @staticmethod
            def json():
                return {"ok": True}

        def _fake_request(method, url, params=None, json=None, timeout=None, headers=None):
            captured["method"] = method
            captured["json"] = json
            return _Response()

        monkeypatch.setattr(tools.httpx, "request", _fake_request)

        tools.simulate_price("H001", "deluxe", "2026-09-15")

        assert captured["method"] == "POST"
        assert captured["json"]["persist"] is False

    def test_overrides_are_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = {}

        class _Response:
            status_code = 200

            @staticmethod
            def json():
                return {}

        monkeypatch.setattr(
            tools.httpx,
            "request",
            lambda method, url, params=None, json=None, timeout=None, headers=None: (
                captured.update(json=json) or _Response()
            ),
        )

        tools.simulate_price(
            "H001", "suite", "2026-09-15", occupancy_rate=0.9, competitor_rate=8000.0
        )
        assert captured["json"]["occupancy_rate"] == 0.9
        assert captured["json"]["competitor_rate"] == 8000.0
        assert captured["json"]["persist"] is False


class TestFailureIsReturnedNotRaised:
    """A tool result the model can read beats an exception that ends the turn."""

    def test_unreachable_api_returns_a_readable_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _refuse(*args, **kwargs):
            raise tools.httpx.ConnectError("connection refused")

        monkeypatch.setattr(tools.httpx, "request", _refuse)
        result = tools.list_hotels()
        assert result.startswith("ERROR")
        assert "could not reach" in result.lower()

    def test_http_error_includes_the_detail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Response:
            status_code = 404
            text = '{"detail": "unknown hotel H999"}'

            @staticmethod
            def json():
                return {"detail": "unknown hotel H999"}

        monkeypatch.setattr(
            tools.httpx, "request", lambda *a, **k: _Response()
        )
        result = tools.get_pricing_history("H999")
        assert "404" in result
        assert "unknown hotel H999" in result

    def test_non_json_error_body_falls_back_to_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A proxy returning an HTML error page must not crash the tool."""
        class _Response:
            status_code = 502
            text = "<html>Bad Gateway</html>"

            @staticmethod
            def json():
                raise ValueError("not json")

        monkeypatch.setattr(tools.httpx, "request", lambda *a, **k: _Response())
        result = tools.list_hotels()
        assert "502" in result
        assert "Bad Gateway" in result

    def test_success_is_compact_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pretty-printing hundreds of competitor rows wastes context window."""
        class _Response:
            status_code = 200

            @staticmethod
            def json():
                return {"hotel_id": "H001", "count": 2}

        monkeypatch.setattr(tools.httpx, "request", lambda *a, **k: _Response())
        result = tools.get_pricing_history("H001")
        assert json.loads(result) == {"hotel_id": "H001", "count": 2}
        assert ", " not in result, "separators should be compact"


class TestOptionalDependency:
    """Everything here must work with no SDK installed."""

    def test_status_reports_what_is_missing(self) -> None:
        status = tools.agent_status()
        assert set(status) >= {
            "available", "sdk_installed", "api_key_present", "tool_count", "problems"
        }
        assert status["tool_count"] == len(tools.RAW_TOOLS)
        if not status["available"]:
            assert status["problems"], "unavailable with no explanation is useless"

    def test_tool_list_length_without_importing_the_sdk(self) -> None:
        """``len(AGENT_TOOLS)`` must not trigger the import.

        The dashboard imports this module on every page load, including when the
        SDK is absent, so it can render install instructions instead of a
        traceback.
        """
        assert len(tools.AGENT_TOOLS) == len(tools.RAW_TOOLS)

    def test_agent_and_triage_import_without_the_sdk(self) -> None:
        import ai_agent.agent
        import ai_agent.triage

        assert ai_agent.agent.SYSTEM_PROMPT
        assert ai_agent.triage.TRIAGE_PROMPT

    def test_build_tools_explains_the_install(self) -> None:
        """Whichever side of the fence this machine is on, the behaviour is right.

        Asserted both ways rather than skipped, so the "no SDK" path -- the one
        most people hit first -- is actually covered on CI as well as the happy
        path on a developer box.
        """
        if tools.agent_status()["sdk_installed"]:
            assert len(tools.build_tools()) == len(tools.RAW_TOOLS)
        else:
            with pytest.raises(tools.AgentUnavailable, match=r"\[agent\]"):
                tools.build_tools()


class TestToolDocstrings:
    """Docstrings are the model's tool-selection prompt, not decoration."""

    def test_every_tool_documents_when_to_call_it(self) -> None:
        for fn in tools.RAW_TOOLS:
            doc = (fn.__doc__ or "").lower()
            assert doc, f"{fn.__name__} has no docstring"
            assert "call this" in doc or "use " in doc, (
                f"{fn.__name__} does not say when to call it; recent models "
                f"reach for tools conservatively and need the trigger stated"
            )

    def test_simulate_price_states_it_cannot_change_prices(self) -> None:
        doc = tools.simulate_price.__doc__ or ""
        assert "cannot change" in doc.lower()

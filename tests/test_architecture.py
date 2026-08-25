"""The modular monolith's boundaries, enforced.

This project is deliberately **not** microservices. At eight hotels and one
operator, splitting it would put network hops inside a 25 ms pricing path, turn
the ``pricing_decisions`` audit trail into a distributed transaction, and
multiply the operational surface by six for no benefit anybody could name.

But "modular monolith" is a claim, and an unenforced claim decays. Every README
that says *"pricing/ imports no framework"* is one careless import away from
being false, and nothing would fail — the code would still run, the tests would
still pass, and the boundary would exist only in prose.

So the boundaries are asserted here, by reading the import graph. These tests
are the difference between a monolith that is modular and a monolith that used
to be.

WHAT EACH RULE BUYS
-------------------
Every rule below exists because breaking it would cost something specific, and
each test names that cost. A rule nobody can justify is a rule that will be
deleted the first time it is inconvenient -- correctly.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Dict, Iterator, List, Set

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Packages that make up the application. Anything else (tests, scripts, the
#: deck generator) is tooling and not part of the layered design.
PACKAGES = [
    "ai_agent",
    "api",
    "config",
    "dashboard",
    "database",
    "demo_ota",
    "domain",
    "features",
    "ingestion",
    "models",
    "monitoring",
    "pricing",
    "streaming",
    "training",
]


def _modules(package: str) -> Iterator[pathlib.Path]:
    """Every Python file in a package, skipping caches."""
    for path in (PROJECT_ROOT / package).rglob("*.py"):
        if "__pycache__" not in path.parts:
            yield path


def _imports(path: pathlib.Path) -> Set[str]:
    """Top-level module names imported by one file.

    Parsed with ``ast`` rather than executed: importing a module to inspect it
    would run its side effects, and ``dashboard/`` modules call
    ``st.set_page_config`` at import time.

    Relative imports are skipped -- they are intra-package by definition and
    cannot cross a boundary.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import
                continue
            if node.module:
                found.add(node.module.split(".")[0])

    return found


@pytest.fixture(scope="module")
def graph() -> Dict[str, Dict[str, Set[str]]]:
    """``{package: {module_path: {imported_top_level_names}}}``."""
    return {
        package: {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _imports(path)
            for path in _modules(package)
        }
        for package in PACKAGES
    }


def _violations(graph, package: str, forbidden: Set[str]) -> List[str]:
    """Every ``module -> forbidden import`` pair inside one package."""
    return [
        f"{module} imports {name!r}"
        for module, names in graph[package].items()
        for name in sorted(names & forbidden)
    ]


# --------------------------------------------------------------------------- #
# The core rule: pricing/ is pure
# --------------------------------------------------------------------------- #


class TestPricingIsPure:
    """``pricing/`` takes numbers and returns numbers."""

    FRAMEWORKS = {
        "fastapi", "starlette", "uvicorn",       # HTTP
        "sqlalchemy", "psycopg2",                 # ORM / driver
        "kafka",                                  # streaming
        "streamlit", "plotly",                    # UI
        "requests", "httpx",                      # network
        "anthropic",                              # LLM
    }

    def test_imports_no_framework(self, graph) -> None:
        """The single most valuable boundary in the project.

        It is what lets every pricing rule be tested in one line with no
        database, no broker and no HTTP client, and it keeps transport and
        persistence concerns out of the module an auditor actually reads.
        Break it and pricing tests start needing fixtures.
        """
        bad = _violations(graph, "pricing", self.FRAMEWORKS)
        assert not bad, "pricing/ must stay framework-free:\n  " + "\n  ".join(bad)

    def test_does_not_import_the_database_layer(self, graph) -> None:
        """Pricing must not know how anything is stored.

        The engine receives a context object; it does not go and fetch one. That
        is what makes the same code path usable from the API, a batch job, and a
        test with hand-built inputs.

        This test is why ``domain/`` exists. It used to fail: ``pricing/``
        imported ``database.models`` for ``RoomType`` and ``Season`` -- plain
        ``str`` enums with no ORM machinery, so nothing broke at runtime, but the
        import graph said pricing depended on persistence. The fix was to move
        the shared vocabulary down a layer rather than to soften the rule into
        something that no longer meant anything.
        """
        bad = _violations(graph, "pricing", {"database"})
        assert not bad, "pricing/ must not reach for persistence:\n  " + "\n  ".join(bad)

    def test_does_not_import_the_api(self, graph) -> None:
        """Dependencies point inward. The API knows about pricing, never the reverse."""
        bad = _violations(graph, "pricing", {"api", "dashboard"})
        assert not bad, "pricing/ must not depend on its callers:\n  " + "\n  ".join(bad)


# --------------------------------------------------------------------------- #
# The dashboard is a pure API consumer
# --------------------------------------------------------------------------- #


class TestDashboardIsAnApiConsumer:
    def test_never_opens_a_database_connection(self, graph) -> None:
        """One implementation of "what is the right price", not two.

        The moment a page can query Postgres directly, a number on screen and
        the number the API would have returned can differ -- and the dashboard
        stops being an integration test of the API.
        """
        bad = _violations(graph, "dashboard", {"database", "sqlalchemy", "psycopg2"})
        assert not bad, "dashboard/ must go through the API:\n  " + "\n  ".join(bad)

    def test_contains_no_pricing_logic(self, graph) -> None:
        """A second implementation of a pricing rule is a second thing to keep correct."""
        bad = _violations(graph, "dashboard", {"pricing", "training"})
        assert not bad, "dashboard/ must not compute prices:\n  " + "\n  ".join(bad)

    def test_does_not_talk_to_kafka(self, graph) -> None:
        bad = _violations(graph, "dashboard", {"kafka", "streaming"})
        assert not bad, "dashboard/ must not publish events:\n  " + "\n  ".join(bad)


# --------------------------------------------------------------------------- #
# The AI agent is contained
# --------------------------------------------------------------------------- #


class TestAgentIsContained:
    """The agent reaches the system the same way any other client does."""

    def test_has_no_database_access(self, graph) -> None:
        """Its read-only guarantee is enforced at the HTTP layer.

        A direct session would route around the allowlist in ``ai_agent/tools.py``
        entirely -- the constraint only holds while HTTP is the *only* way out.
        """
        bad = _violations(graph, "ai_agent", {"database", "sqlalchemy", "psycopg2"})
        assert not bad, "ai_agent/ must not touch the database:\n  " + "\n  ".join(bad)

    def test_cannot_call_the_pricing_engine_directly(self, graph) -> None:
        """It must simulate through the API, where persist=False is enforced.

        Calling PricingEngine in-process would produce a price that never passed
        through the endpoint -- and therefore never through the audit trail.
        """
        bad = _violations(graph, "ai_agent", {"pricing", "training", "features"})
        assert not bad, "ai_agent/ must go through the API:\n  " + "\n  ".join(bad)

    def test_nothing_in_the_system_imports_the_agent(self, graph) -> None:
        """The dependency direction that makes it deletable.

        Only the dashboard page may import it. If the API, the pricing engine or
        the models did, then removing an optional LLM dependency would break the
        system that is supposed to work without one.
        """
        offenders = [
            f"{module} imports 'ai_agent'"
            for package in PACKAGES
            if package != "ai_agent"
            for module, names in graph[package].items()
            if "ai_agent" in names and not module.startswith("dashboard/pages/")
        ]
        assert not offenders, (
            "ai_agent/ must stay a leaf -- nothing may depend on it:\n  "
            + "\n  ".join(offenders)
        )

    def test_the_llm_sdk_is_confined_to_the_agent(self, graph) -> None:
        """``anthropic`` must appear in exactly one package.

        It is an optional extra. An import anywhere else turns "the system runs
        without an LLM SDK" from a tested property into a hope.
        """
        offenders = [
            f"{module} imports 'anthropic'"
            for package in PACKAGES
            if package != "ai_agent"
            for module, names in graph[package].items()
            if "anthropic" in names
        ]
        assert not offenders, "anthropic must stay inside ai_agent/:\n  " + "\n  ".join(offenders)


# --------------------------------------------------------------------------- #
# The demo OTA is a third party
# --------------------------------------------------------------------------- #


class TestDemoOtaIsAThirdParty:
    """It stands in for somebody else's website, so it must behave like one."""

    def test_imports_nothing_from_the_application(self, graph) -> None:
        """The scraper reaches it over HTTP, never by import.

        If ``demo_ota`` could import ``config`` or ``database``, the scraping
        demo would be measuring a function call dressed up as a network request.
        """
        internal = set(PACKAGES) - {"demo_ota"}
        bad = _violations(graph, "demo_ota", internal)
        assert not bad, (
            "demo_ota/ must stay independent -- it is a stand-in for a "
            "third-party site:\n  " + "\n  ".join(bad)
        )

    def test_the_scrapers_do_not_import_it(self, graph) -> None:
        """``ingestion/`` must reach it the same way it would reach Booking.com."""
        bad = _violations(graph, "ingestion", {"demo_ota"})
        assert not bad, (
            "ingestion/ must scrape demo_ota over HTTP, not import it:\n  "
            + "\n  ".join(bad)
        )


# --------------------------------------------------------------------------- #
# Layering
# --------------------------------------------------------------------------- #


class TestLayering:
    """Dependencies point inward: config <- database <- features/models <- pricing <- api."""

    #: What each package is forbidden from importing, by layer position.
    RULES = {
        "domain": {"api", "dashboard", "pricing", "features", "models", "training",
                   "streaming", "ingestion", "monitoring", "database", "ai_agent",
                   "config"},
        "config": {"api", "dashboard", "pricing", "features", "models", "training",
                   "streaming", "ingestion", "monitoring", "database", "ai_agent"},
        "database": {"api", "dashboard", "pricing", "features", "models", "training",
                     "ingestion", "ai_agent"},
        "features": {"api", "dashboard", "training", "ai_agent"},
        "models": {"api", "dashboard", "ai_agent"},
        "ingestion": {"api", "dashboard", "pricing", "training", "ai_agent"},
        "streaming": {"api", "dashboard", "pricing", "training", "ai_agent"},
        "training": {"api", "dashboard", "ai_agent"},
        "monitoring": {"api", "dashboard", "ai_agent"},
    }

    @pytest.mark.parametrize("package", sorted(RULES))
    def test_package_does_not_import_upward(self, graph, package: str) -> None:
        bad = _violations(graph, package, self.RULES[package])
        assert not bad, (
            f"{package}/ imports from a higher layer -- dependencies point "
            f"inward:\n  " + "\n  ".join(bad)
        )

    def test_config_is_a_leaf(self, graph) -> None:
        """Everything reads configuration; configuration reads nothing.

        A cycle here means importing any module can trigger settings validation
        at an arbitrary point in startup.
        """
        bad = _violations(graph, "config", set(PACKAGES) - {"config"})
        assert not bad, "config/ must depend on nothing internal:\n  " + "\n  ".join(bad)


# --------------------------------------------------------------------------- #
# Guardrails against the specific mistakes this project has already made
# --------------------------------------------------------------------------- #


class TestKnownFootguns:
    def test_only_guardrails_constructs_a_final_price(self) -> None:
        """``FinalPrice`` must remain unconstructable outside the guardrail module.

        The token makes "every served price passed the guardrails" a property of
        the type system. A second construction site would quietly turn it back
        into a convention.
        """
        sites = [
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            for package in PACKAGES
            for path in _modules(package)
            if "_CONSTRUCTION_TOKEN" in path.read_text(encoding="utf-8")
        ]
        assert sites == ["pricing/guardrails.py"], (
            "the FinalPrice construction token must exist in exactly one module, "
            f"found: {sites}"
        )

    def test_no_module_reads_the_clock_at_import_time(self) -> None:
        """A module-level ``date.today()`` freezes at import.

        A long-running process would then keep pricing the dates it was started
        with -- exactly the bug the producer had, where every sweep re-scraped
        the check-in dates the process booted with.

        ``dashboard/`` is exempt, and the exemption is real rather than
        convenient: Streamlit re-executes a page's entire module on every widget
        interaction, so "import time" there *is* request time. The same line that
        would be frozen in a long-lived service is re-evaluated per rerun.
        """
        offenders = []
        for package in PACKAGES:
            if package == "dashboard":
                continue
            for path in _modules(package):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in tree.body:  # top level only
                    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                        continue
                    for call in ast.walk(node):
                        if not isinstance(call, ast.Call):
                            continue
                        target = getattr(call.func, "attr", None)
                        if target in {"today", "now", "utcnow"}:
                            offenders.append(
                                f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} "
                                f"calls {target}() at import time"
                            )
        assert not offenders, "\n  " + "\n  ".join(offenders)

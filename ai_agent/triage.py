"""Turns a monitoring report into an ops note a human can act on.

``scripts/monitor.py`` produces ``data/monitoring_report.json``: nine data-quality
checks, PSI drift per feature, prediction statistics, each with a severity. On a
bad night six of them go amber and five are the same root cause. Alert fatigue is
not a threshold problem -- no threshold can say "these three are downstream of
that one" -- it is a summarisation problem, which is what this module is for.

Batch, not interactive. Run it after the monitor, nightly; nobody is waiting, so
the latency is free and the Batches API would halve the cost if volume ever
justified it.

Read-only in the strongest sense: it takes a JSON file and returns a string. It
holds no tools at all, so unlike the agent there is nothing to constrain.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ai_agent.tools import AgentUnavailable, _load_anthropic
from config import get_settings
from monitoring.logging_config import get_logger

logger = get_logger(__name__)

MODEL = os.getenv("AGENT_MODEL", "claude-opus-4-8")

#: The report is a few thousand tokens and the output is a short note.
MAX_TOKENS = 4_000

TRIAGE_PROMPT = """\
You are the on-call engineer for a dynamic hotel pricing system. You are handed
the nightly monitoring report and you write the note the morning shift reads.

WHAT THE SYSTEM IS
Competitor rates arrive by scraper into Kafka, a consumer writes them to
PostgreSQL, a feature pipeline builds a 30-feature matrix, and two models
(Prophet for demand seasonality, a gradient booster for the point estimate) feed
a deterministic pricing engine.

DOMAIN FACTS THAT CHANGE HOW YOU READ THIS REPORT
- Drift and seasonality are genuinely confounded here. Indian hotel demand is
  strongly seasonal, so a PSI shift on occupancy or season features in late
  October is what the calendar looks like, not evidence the model has decayed.
  The report carries an explicit seasonality caveat -- respect it, and say so.
- Competitor coverage below about 70% almost always means the producer or the
  scraper stopped, not that the market went quiet. Check whether the other
  competitor-related signals moved at the same time; if they did, that is one
  incident, not four.
- Prediction variance collapsing towards zero usually means the feed died and
  every row is being scored on identical stale features.
- A small sample makes every statistic look alarming. If a check fired on a
  handful of predictions, say the sample is too small to act on.

HOW TO WRITE THE NOTE
1. One sentence: is anything actually wrong, and does it need action today?
2. Group the alerts by root cause. Name explicitly which ones are downstream
   symptoms of another, and which are independent.
3. For each real issue: what to check first, with the specific command, table or
   service to look at.
4. A short "safe to ignore tonight" list, each with the reason.

Be concise and specific. Do not restate every check that passed. If the report is
clean, say so in two sentences and stop -- padding a quiet night with speculation
is how people learn to skip the note.
"""


@dataclass
class TriageNote:
    """A generated ops note plus what it cost."""

    note: str
    severity: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def estimated_cost_usd(self) -> float:
        return self.input_tokens / 1e6 * 5.0 + self.output_tokens / 1e6 * 25.0


def report_path() -> Path:
    """Where ``scripts/monitor.py`` writes its report."""
    return get_settings().paths.data_dir / "monitoring_report.json"


def load_report(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read the monitoring report.

    Raises:
        FileNotFoundError: If no report exists. Run ``make monitor`` first --
            triage summarises a report, it does not run the checks itself.
    """
    target = path or report_path()
    if not target.is_file():
        raise FileNotFoundError(
            f"No monitoring report at {target}. Run 'make monitor' (or "
            f"'python scripts/monitor.py') to produce one."
        )
    return json.loads(target.read_text(encoding="utf-8"))


def triage(report: Optional[Dict[str, Any]] = None, *, path: Optional[Path] = None) -> TriageNote:
    """Summarise a monitoring report into an actionable note.

    Args:
        report: An already-loaded report. Loaded from disk when omitted.
        path: Where to load from, when ``report`` is not given.

    Raises:
        AgentUnavailable: If the SDK or the API key is missing.
        FileNotFoundError: If no report was given and none exists on disk.
    """
    anthropic = _load_anthropic()
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        raise AgentUnavailable("No ANTHROPIC_API_KEY in the environment.")

    payload = report if report is not None else load_report(path)
    severity = str(payload.get("severity", "unknown"))

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": TRIAGE_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Here is tonight's monitoring report. Write the note.\n\n"
                    f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"
                ),
            }
        ],
    )

    note = "\n".join(b.text for b in response.content if b.type == "text").strip()

    return TriageNote(
        note=note,
        severity=severity,
        input_tokens=response.usage.input_tokens or 0,
        output_tokens=response.usage.output_tokens or 0,
    )


def main(argv: Optional[list] = None) -> int:
    """CLI entry point: read the latest report, print a note.

    Exits non-zero only when triage could not run. A bad report is a successful
    triage -- the caller wanted the note, and got it.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m ai_agent.triage",
        description="Summarise the latest monitoring report into an ops note.",
    )
    parser.add_argument("--report", type=Path, default=None, help="Path to a report JSON.")
    parser.add_argument("--out", type=Path, default=None, help="Also write the note here.")
    args = parser.parse_args(argv)

    try:
        result = triage(path=args.report)
    except (AgentUnavailable, FileNotFoundError) as exc:
        logger.error("%s", exc)
        print(f"\nCannot run triage: {exc}\n")
        return 1

    print()
    print("=" * 70)
    print(f"MONITORING TRIAGE  --  overall severity: {result.severity.upper()}")
    print("=" * 70)
    print(result.note)
    print("=" * 70)
    print(
        f"  {result.input_tokens:,} in / {result.output_tokens:,} out "
        f"(~${result.estimated_cost_usd:.3f})"
    )
    print()

    if args.out:
        args.out.write_text(result.note, encoding="utf-8")
        logger.info("Wrote the note to %s", args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["TRIAGE_PROMPT", "TriageNote", "load_report", "report_path", "triage"]

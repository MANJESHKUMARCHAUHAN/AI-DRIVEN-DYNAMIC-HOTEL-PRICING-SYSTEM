"""Run the monitoring checks and write a report.

Data quality first, model health second -- in that order deliberately. Most
production ML failures are data failures wearing a model's clothes, and a drift
number computed over a broken feature build is a distraction.

Usage::

    python scripts/monitor.py                 # run everything, print a report
    python scripts/monitor.py --json          # machine-readable, for a cron job
    python scripts/monitor.py --data-only
    python scripts/monitor.py --fail-on warning

The exit code is the point when this runs on a schedule: ``--fail-on`` turns a
severity into a non-zero exit, which is what a cron wrapper or a CI step keys
off.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow execution as a plain script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings  # noqa: E402
from database.connection import session_scope, wait_for_database  # noqa: E402
from monitoring.data_monitor import DataMonitor, Severity  # noqa: E402
from monitoring.logging_config import configure_logging, get_logger  # noqa: E402
from monitoring.model_monitor import ModelMonitor  # noqa: E402

logger = get_logger(__name__)

#: Where the dashboard looks for the latest report.
REPORT_FILENAME = "monitoring_report.json"

_SEVERITY_ORDER = {Severity.OK: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}

_MARKER = {Severity.OK: "  ok  ", Severity.WARNING: " WARN ", Severity.CRITICAL: " CRIT "}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/monitor.py",
        description="Run data quality and model health monitoring.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    parser.add_argument("--data-only", action="store_true", help="Skip model checks.")
    parser.add_argument("--model-only", action="store_true", help="Skip data checks.")
    parser.add_argument(
        "--fail-on",
        choices=["never", "warning", "critical"],
        default="never",
        help="Exit non-zero at this severity or worse.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the JSON report. Defaults to DATA_DIR.",
    )
    return parser


def _print_report(payload: Dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print("MONITORING REPORT")
    print("=" * 78)
    print(f"  generated {payload['generated_at']}")
    print(f"  severity  {payload['severity'].upper()}")

    data = payload.get("data_quality")
    if data:
        print("\n" + "-" * 78)
        print(f"DATA QUALITY -- {data['passed']}/{data['total']} passed")
        print("-" * 78)
        for check in data["checks"]:
            marker = _MARKER[Severity(check["severity"])]
            print(f" [{marker}] {check['name']:<24} {check['message']}")

    model = payload.get("model_health")
    if model:
        print("\n" + "-" * 78)
        print("MODEL HEALTH")
        print("-" * 78)
        for check in model["checks"]:
            marker = _MARKER[Severity(check["severity"])]
            print(f" [{marker}] {check['name']:<24} {check['message']}")

        if model["drift"]:
            print("\n  feature drift (PSI vs the training window):")
            for entry in sorted(
                model["drift"], key=lambda d: d["psi"] or 0, reverse=True
            ):
                psi = entry["psi"]
                marker = _MARKER[Severity(entry["severity"])]
                value = f"{psi:.3f}" if psi is not None else "  n/a"
                print(
                    f"   [{marker}] {entry['feature']:<22} PSI {value}"
                    f"   mean {entry['reference_mean']:.3f} -> {entry['current_mean']:.3f}"
                )
        else:
            print("\n  feature drift: not enough history to measure")

        if model.get("guardrail_counts"):
            print("\n  guardrails fired:")
            for rule, count in sorted(
                model["guardrail_counts"].items(), key=lambda kv: -kv[1]
            ):
                print(f"    {rule:<28} {count:>6}")

        if model.get("accuracy"):
            accuracy = model["accuracy"]
            print(
                f"\n  realised accuracy on completed nights: "
                f"MAE {accuracy['mae']:.4f}  R2 {accuracy['r2']:.3f}  n={accuracy['n']}"
            )

    print("=" * 78)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    settings = get_settings()
    configure_logging(settings)
    args = _build_parser().parse_args(argv)

    if not settings.monitoring.enabled:
        logger.warning("MONITORING_ENABLED=false; nothing to do")
        return 0

    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "severity": Severity.OK.value,
    }
    worst = Severity.OK

    try:
        wait_for_database()
        with session_scope() as session:
            if not args.model_only:
                report = DataMonitor(settings).run(session)
                payload["data_quality"] = report.as_dict()
                worst = max(worst, report.severity, key=lambda s: _SEVERITY_ORDER[s])

            if not args.data_only:
                health = ModelMonitor(settings).run(session)
                payload["model_health"] = health.as_dict()
                worst = max(worst, health.severity, key=lambda s: _SEVERITY_ORDER[s])
    except Exception as exc:
        logger.error("Monitoring failed: %s: %s", type(exc).__name__, exc)
        return 1

    payload["severity"] = worst.value

    output_dir = args.output or settings.paths.data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / REPORT_FILENAME
    report_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        _print_report(payload)
        print(f"\nWritten to {report_path}")

    thresholds = {
        "never": None,
        "warning": Severity.WARNING,
        "critical": Severity.CRITICAL,
    }
    limit = thresholds[args.fail_on]
    if limit is not None and _SEVERITY_ORDER[worst] >= _SEVERITY_ORDER[limit]:
        logger.error("Monitoring severity %s meets --fail-on %s", worst.value, args.fail_on)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

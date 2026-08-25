"""Configuration doctor.

Loads the full settings tree, prints a redacted summary and reports any
validation failure with an actionable message. Run it after editing ``.env`` and
before starting containers::

    python scripts/check_config.py

Exit code 0 means the configuration is valid; 1 means it is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow execution as a plain script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError  # noqa: E402

from config import ENV_FILE, PROJECT_ROOT, get_settings  # noqa: E402


def main() -> int:
    """Validate configuration and report the outcome."""
    print("=" * 78)
    print("CONFIGURATION CHECK")
    print("=" * 78)
    print(f"project root : {PROJECT_ROOT}")
    print(f"env file     : {ENV_FILE}")
    print(f"env exists   : {Path(ENV_FILE).exists()}")
    print("-" * 78)

    try:
        settings = get_settings()
    except ValidationError as exc:
        print("INVALID CONFIGURATION\n")
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"]) or "(root)"
            print(f"  {location}: {error['msg']}")
        print("\nFix the values above in your .env file and re-run.")
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        print(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}")
        return 1

    for key, value in settings.summary().items():
        print(f"  {key:<24} : {value}")

    print("-" * 78)

    settings.ensure_directories()
    print("Writable directories:")
    for label, path in (
        ("data", settings.paths.data_dir),
        ("data/raw", settings.paths.raw_dir),
        ("data/processed", settings.paths.processed_dir),
        ("data/synthetic", settings.paths.synthetic_dir),
        ("model artifacts", settings.model.model_dir),
    ):
        exists = "ok" if path.exists() else "MISSING"
        print(f"  {label:<24} : {path}  [{exists}]")

    print("-" * 78)
    print("Configuration is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

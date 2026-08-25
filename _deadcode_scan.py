"""Find definitions that nothing references.

Parses every module with ``ast``, collects top-level functions and classes, then
counts how many times each name appears as a *usage* anywhere in the project
(including tests and docs). Anything with zero usages outside its own definition
is a candidate.

Candidates, not verdicts. Dunder methods, FastAPI route handlers, pytest fixtures
and anything re-exported through ``__all__`` are legitimately "uncalled" and are
filtered out here rather than reported as noise.
"""

from __future__ import annotations

import ast
import pathlib
import re
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent

PACKAGES = [
    "ai_agent", "api", "config", "dashboard", "database", "demo_ota", "domain",
    "features", "ingestion", "models", "monitoring", "pricing", "scripts",
    "streaming", "training",
]

SEARCHED = PACKAGES + ["tests", "docs"]


def modules():
    for package in PACKAGES:
        for path in (ROOT / package).rglob("*.py"):
            if "__pycache__" not in path.parts:
                yield path


def all_text() -> str:
    """Every file a name could be referenced from."""
    chunks = []
    for area in SEARCHED:
        for path in (ROOT / area).rglob("*"):
            if path.suffix in {".py", ".md"} and "__pycache__" not in path.parts:
                chunks.append(path.read_text(encoding="utf-8"))
    for extra in ("README.md", "Makefile", "docker-compose.yml", "pyproject.toml"):
        p = ROOT / extra
        if p.exists():
            chunks.append(p.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def main() -> None:
    corpus = all_text()
    counts = defaultdict(int)

    definitions = []  # (name, kind, path, lineno, is_exported, decorators)

    for path in modules():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))

        exported: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if getattr(target, "id", None) == "__all__":
                        exported = {
                            el.value for el in getattr(node.value, "elts", [])
                            if isinstance(el, ast.Constant)
                        }

        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            name = node.name
            decorators = [
                ast.unparse(d) if hasattr(ast, "unparse") else "" for d in node.decorator_list
            ]
            definitions.append(
                (name, type(node).__name__, path.relative_to(ROOT), node.lineno,
                 name in exported, decorators)
            )

    for name, *_ in definitions:
        # Count bare-word occurrences; the definition line itself is one of them.
        counts[name] = len(re.findall(rf"\b{re.escape(name)}\b", corpus))

    print("CANDIDATE DEAD DEFINITIONS (referenced only where defined)\n")
    found = 0
    for name, kind, path, lineno, exported, decorators in sorted(definitions, key=lambda d: str(d[2])):
        if name.startswith("__") and name.endswith("__"):
            continue
        if exported:
            continue
        # Route handlers, fixtures, validators and CLI entry points are invoked
        # by a framework, never by name.
        if any(
            key in d
            for d in decorators
            for key in ("router.", "app.", "fixture", "validator", "computed_field",
                        "beta_tool", "property", "parametrize", "staticmethod")
        ):
            continue
        if counts[name] <= 1:
            print(f"  {str(path):46} :{lineno:<5} {kind[:5]:6} {name}")
            found += 1

    print(f"\n{found} candidate(s) out of {len(definitions)} definitions")


if __name__ == "__main__":
    main()

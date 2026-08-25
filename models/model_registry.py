"""The model registry: which artifacts exist, which one is live, is it loadable.

Backed by the filesystem. A training run writes ``gbr_<version>.joblib``,
``prophet_<version>.joblib`` and ``training_report_<version>.json`` into the
artifact directory; this module reads that directory back and turns it into
something the API can serve from (ADR-005).

Three properties the API depends on:

**Loading is lazy and never fatal.** A missing artifact is a degraded service,
not a failed boot. The API starts, ``/health`` says the models are absent, and
``/pricing/predict`` still answers using whichever model did load -- or the
historical fallback if neither did. An API that refuses to start because a
training job has not run yet is an API that cannot be deployed before it is
trained, which is a deployment order nobody wants to be forced into.

**The feature contract is checked at load, not at first request.** Every
artifact is saved beside the ordered feature list it was trained on. If the
running code produces a different list, loading fails loudly here rather than
producing quietly wrong prices for a week.

**Reload is explicit and atomic.** :meth:`ModelRegistry.reload` builds the new
bundle fully before swapping it in, so a failed reload leaves the previous
models serving rather than leaving the process with none.

Phase 10 adds database registration and drift monitoring on top of this; the
file layer is deliberately independent so the API keeps working when the
database does not.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import Settings, get_settings
from features.feature_store import (
    FeatureVersionMismatch,
    load_feature_list,
    validate_feature_list,
)
from monitoring.logging_config import get_logger
from monitoring.metrics import set_model_version

logger = get_logger(__name__)

#: Artifact filename patterns. Version is the capture group.
_GBR_PATTERN = re.compile(r"^gbr_(v\d+)\.joblib$")
_PROPHET_PATTERN = re.compile(r"^prophet_(v\d+)\.joblib$")
_REPORT_PATTERN = re.compile(r"^training_report_(v\d+)\.json$")


def _version_number(version: str) -> int:
    """``"v12"`` -> ``12``, for ordering."""
    match = re.fullmatch(r"v(\d+)", version)
    return int(match.group(1)) if match else -1


@dataclass
class ModelVersionInfo:
    """What exists on disk for one version."""

    version: str
    gbr_path: Optional[Path] = None
    prophet_path: Optional[Path] = None
    report_path: Optional[Path] = None
    report: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_any(self) -> bool:
        return self.gbr_path is not None or self.prophet_path is not None

    @property
    def trained_at(self) -> Optional[str]:
        return self.report.get("finished_at") or self.report.get("started_at")

    def metrics_for(self, step: str) -> Dict[str, Any]:
        for entry in self.report.get("steps", []):
            if entry.get("name") == step and entry.get("succeeded"):
                return entry.get("metrics", {})
        return {}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "gradient_boosting": self.gbr_path.name if self.gbr_path else None,
            "prophet": self.prophet_path.name if self.prophet_path else None,
            "trained_at": self.trained_at,
            "feature_version": self.report.get("feature_version"),
            "dataset_hash": self.report.get("dataset_hash"),
            "n_train": self.report.get("n_train"),
            "n_test": self.report.get("n_test"),
            "metrics": {
                "gradient_boosting": self.metrics_for("gradient_boosting"),
                "prophet": self.metrics_for("prophet"),
            },
        }


@dataclass
class LoadedModels:
    """The models currently serving, and how loading went.

    ``errors`` is populated rather than raised: a Prophet artifact that fails to
    load should not stop the Gradient Boosting model from serving.
    """

    version: Optional[str] = None
    gbr: Optional[Any] = None
    prophet: Optional[Any] = None
    loaded_at: Optional[datetime] = None
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def any_loaded(self) -> bool:
        return self.gbr is not None or self.prophet is not None

    @property
    def available(self) -> List[str]:
        return [
            name
            for name, model in (("gradient_boosting", self.gbr), ("prophet", self.prophet))
            if model is not None
        ]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "active_version": self.version,
            "available": self.available,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "errors": dict(self.errors),
        }


class ModelRegistry:
    """Discovers, loads and serves the trained model artifacts.

    Example::

        registry = ModelRegistry()
        registry.load()
        registry.loaded.gbr.predict(frame)
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.artifact_dir: Path = self.settings.model.model_dir
        self._loaded = LoadedModels()
        self._lock = threading.Lock()

    # -- discovery ----------------------------------------------------------- #

    def discover(self) -> Dict[str, ModelVersionInfo]:
        """Every version present on disk, newest last."""
        if not self.artifact_dir.is_dir():
            return {}

        versions: Dict[str, ModelVersionInfo] = {}

        def slot(version: str) -> ModelVersionInfo:
            return versions.setdefault(version, ModelVersionInfo(version=version))

        for path in sorted(self.artifact_dir.iterdir()):
            if match := _GBR_PATTERN.match(path.name):
                slot(match.group(1)).gbr_path = path
            elif match := _PROPHET_PATTERN.match(path.name):
                slot(match.group(1)).prophet_path = path
            elif match := _REPORT_PATTERN.match(path.name):
                info = slot(match.group(1))
                info.report_path = path
                try:
                    info.report = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("Could not read %s: %s", path.name, exc)

        return dict(
            sorted(versions.items(), key=lambda item: _version_number(item[0]))
        )

    def latest_version(self) -> Optional[str]:
        """The newest version that has at least one loadable artifact."""
        usable = [v for v, info in self.discover().items() if info.has_any]
        return max(usable, key=_version_number) if usable else None

    def resolve_version(self, requested: Optional[str] = None) -> Optional[str]:
        """Which version to serve.

        Precedence: the explicit argument, then ``MODEL_ACTIVE_VERSION``, then
        the newest on disk. Pinning matters -- "whatever trained last" is fine
        for a laptop and unacceptable in production.
        """
        if requested:
            return requested
        configured = self.settings.model.model_active_version
        if configured:
            return configured
        return self.latest_version()

    # -- loading -------------------------------------------------------------- #

    @property
    def loaded(self) -> LoadedModels:
        return self._loaded

    @property
    def is_loaded(self) -> bool:
        return self._loaded.any_loaded

    def load(self, version: Optional[str] = None) -> LoadedModels:
        """Load a version's artifacts, replacing whatever is live.

        Never raises for a missing or broken artifact -- the failure is recorded
        on the returned :class:`LoadedModels` and the API degrades. A feature
        contract mismatch *is* recorded too, and is the one worth alerting on.
        """
        target = self.resolve_version(version)
        if target is None:
            logger.warning(
                "No model artifacts found in %s; pricing will run on the "
                "historical fallback until scripts/train_models.py has been run",
                self.artifact_dir,
            )
            with self._lock:
                self._loaded = LoadedModels(errors={"registry": "no artifacts found"})
            return self._loaded

        info = self.discover().get(target)
        if info is None or not info.has_any:
            message = f"version {target!r} has no artifacts in {self.artifact_dir}"
            logger.error("%s", message)
            with self._lock:
                self._loaded = LoadedModels(errors={"registry": message})
            return self._loaded

        errors: Dict[str, str] = {}

        # The train/serve contract, checked once at load rather than per request.
        try:
            validate_feature_list(load_feature_list(self.artifact_dir))
        except FileNotFoundError as exc:
            errors["feature_list"] = str(exc)
            logger.warning("No feature list beside the artifacts: %s", exc)
        except FeatureVersionMismatch as exc:
            errors["feature_list"] = str(exc)
            logger.error(
                "FEATURE CONTRACT MISMATCH -- serving these models would produce "
                "silently wrong prices: %s",
                exc,
            )

        gbr = self._load_gbr(info, errors)
        prophet = self._load_prophet(info, errors)

        loaded = LoadedModels(
            version=target,
            gbr=gbr,
            prophet=prophet,
            loaded_at=datetime.now(timezone.utc),
            errors=errors,
        )

        if loaded.any_loaded:
            logger.info(
                "Model registry loaded %s: %s", target, ", ".join(loaded.available)
            )
            # Publish which artifact is actually serving. `model_version_info` is
            # documented as an available metric, and a declared metric that
            # nothing ever sets is a lie on a dashboard rather than a gap.
            for name in loaded.available:
                set_model_version(model=name, version=loaded.version or "unversioned")
        else:
            logger.error("Model registry could not load any model for %s", target)

        # Swapped in only once fully built, so a failed reload leaves the
        # previous models serving.
        with self._lock:
            self._loaded = loaded
        return loaded

    def _load_gbr(self, info: ModelVersionInfo, errors: Dict[str, str]) -> Optional[Any]:
        if info.gbr_path is None:
            errors["gradient_boosting"] = "artifact not present"
            return None
        try:
            from models.gradient_boosting_model import GradientBoostingDemandModel

            model = GradientBoostingDemandModel.load(info.gbr_path)
            if "feature_list" in errors:
                # The contract already failed; refuse to serve rather than
                # trusting a model whose inputs we know have moved.
                errors["gradient_boosting"] = (
                    "not served: the feature contract does not match this artifact"
                )
                return None
            return model
        except Exception as exc:
            errors["gradient_boosting"] = f"{type(exc).__name__}: {exc}"
            logger.error("Failed to load %s: %s", info.gbr_path.name, exc)
            return None

    def _load_prophet(
        self, info: ModelVersionInfo, errors: Dict[str, str]
    ) -> Optional[Any]:
        if info.prophet_path is None:
            errors["prophet"] = "artifact not present"
            return None
        try:
            from models.prophet_model import ProphetBundle

            # Prophet reads only dates, so a feature-list change does not
            # invalidate it the way it invalidates the GBR.
            return ProphetBundle.load(info.prophet_path)
        except Exception as exc:
            errors["prophet"] = f"{type(exc).__name__}: {exc}"
            logger.error("Failed to load %s: %s", info.prophet_path.name, exc)
            return None

    def reload(self, version: Optional[str] = None) -> LoadedModels:
        """Re-read artifacts from disk. Used after a training run."""
        logger.info("Reloading model registry from %s", self.artifact_dir)
        return self.load(version)

    def ensure_loaded(self) -> LoadedModels:
        """Load on first use. Cheap to call repeatedly."""
        if not self._loaded.any_loaded and not self._loaded.errors:
            self.load()
        return self._loaded

    # -- reporting ------------------------------------------------------------ #

    def catalogue(self) -> List[Dict[str, Any]]:
        """Every version on disk, newest first, with its metrics.

        Backs ``GET /api/v1/models``.
        """
        active = self._loaded.version
        rows = [
            {**info.as_dict(), "is_active": info.version == active}
            for info in self.discover().values()
        ]
        return sorted(rows, key=lambda r: _version_number(r["version"]), reverse=True)

    def status(self) -> Dict[str, Any]:
        """Compact status for ``/health`` and the monitoring page."""
        return {
            **self._loaded.as_dict(),
            "artifact_dir": str(self.artifact_dir),
            "versions_on_disk": list(self.discover()),
        }


#: Process-wide registry. Models are large and loading them per request would
#: dominate response time; the API loads once at startup and reloads on demand.
_registry: Optional[ModelRegistry] = None
_registry_lock = threading.Lock()


def get_registry(settings: Optional[Settings] = None) -> ModelRegistry:
    """Return the process-wide registry, creating it on first call."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ModelRegistry(settings)
    return _registry


def reset_registry() -> None:
    """Drop the process-wide registry. For tests and for a hard reload."""
    global _registry
    with _registry_lock:
        _registry = None


__all__ = [
    "LoadedModels",
    "ModelRegistry",
    "ModelVersionInfo",
    "get_registry",
    "reset_registry",
]

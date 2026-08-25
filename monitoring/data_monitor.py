"""Data quality monitoring: is the input to the models still trustworthy?

Model monitoring gets the attention, but most production ML failures are data
failures wearing a model's clothes. A competitor feed that quietly stops, a
feature pipeline that has not run for two days, a column that starts arriving
null -- none of these raise an exception, and all of them make the prices wrong
while every model metric looks fine.

Each check returns a :class:`CheckResult` with a severity rather than raising.
The point is a report you can look at, not an exception that stops a batch job:
"three of nine checks are warning" is actionable, and a traceback on the first
problem hides the other two.

Severity means something specific here:

``OK``
    Nothing to do.
``WARNING``
    Prices are still safe to serve, but somebody should look. Stale competitor
    data, a feature run that is a day behind.
``CRITICAL``
    The inputs can no longer be trusted. An empty feature store, a table with no
    rows for today, nulls in a column the model requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from config import Settings, get_settings
from database.models import Booking, CompetitorPrice, DemandFeature, Hotel, Room
from features.feature_engineering import FEATURE_VERSION
from monitoring.logging_config import get_logger

logger = get_logger(__name__)

#: Competitor observations older than this mean the market view has gone stale.
COMPETITOR_STALENESS_HOURS = 48

#: The feature store should be rebuilt at least this often.
FEATURE_STALENESS_HOURS = 36

#: Below this share of nights having any competitor rate, the competitor
#: features are mostly imputed and the competitor adjustment is mostly zero.
MIN_COMPETITOR_COVERAGE = 0.50


class Severity(str, Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CheckResult:
    """The outcome of one data quality check."""

    name: str
    severity: Severity
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.severity is Severity.OK

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "detail": self.detail,
        }

    def log(self) -> None:
        if self.severity is Severity.CRITICAL:
            logger.error("DATA CHECK %s: %s", self.name, self.message)
        elif self.severity is Severity.WARNING:
            logger.warning("DATA CHECK %s: %s", self.name, self.message)
        else:
            logger.info("DATA CHECK %s: %s", self.name, self.message)


@dataclass
class DataQualityReport:
    """Every check, plus the worst severity across them."""

    checks: List[CheckResult] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def severity(self) -> Severity:
        if any(c.severity is Severity.CRITICAL for c in self.checks):
            return Severity.CRITICAL
        if any(c.severity is Severity.WARNING for c in self.checks):
            return Severity.WARNING
        return Severity.OK

    @property
    def failures(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "severity": self.severity.value,
            "passed": sum(1 for c in self.checks if c.passed),
            "total": len(self.checks),
            "checks": [c.as_dict() for c in self.checks],
        }

    def summary(self) -> str:
        passed = sum(1 for c in self.checks if c.passed)
        return f"{passed}/{len(self.checks)} checks passed, worst severity {self.severity.value}"


class DataMonitor:
    """Runs the data quality checks against the live database.

    Example::

        with session_scope() as session:
            report = DataMonitor().run(session)
            print(report.summary())
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    # -- individual checks ---------------------------------------------------- #

    @staticmethod
    def check_reference_data(session: Session) -> CheckResult:
        """Hotels and rooms exist, and every hotel has rooms."""
        hotels = session.execute(select(func.count()).select_from(Hotel.__table__)).scalar_one()
        rooms = session.execute(select(func.count()).select_from(Room.__table__)).scalar_one()

        if hotels == 0:
            return CheckResult(
                "reference_data",
                Severity.CRITICAL,
                "no hotels are configured; nothing can be priced",
                value=0,
            )

        orphans = session.execute(
            select(func.count())
            .select_from(Hotel.__table__)
            .where(~Hotel.hotel_id.in_(select(Room.hotel_id)))
        ).scalar_one()

        if orphans:
            return CheckResult(
                "reference_data",
                Severity.CRITICAL,
                f"{orphans} hotel(s) have no rooms and cannot be priced",
                value=float(orphans),
            )

        return CheckResult(
            "reference_data",
            Severity.OK,
            f"{hotels} hotel(s), {rooms} room type(s)",
            value=float(hotels),
        )

    @staticmethod
    def check_booking_recency(session: Session) -> CheckResult:
        """Bookings are still arriving.

        A booking table that stops growing is either a broken pipeline or a
        hotel that has stopped selling. Both need someone to look.
        """
        latest = session.execute(select(func.max(Booking.check_in_date))).scalar_one_or_none()
        if latest is None:
            return CheckResult(
                "booking_recency", Severity.CRITICAL, "the bookings table is empty"
            )

        age = (date.today() - latest).days
        if age > 7:
            return CheckResult(
                "booking_recency",
                Severity.WARNING,
                f"the most recent stay date is {age} days old ({latest})",
                value=float(age),
                threshold=7,
            )
        return CheckResult(
            "booking_recency",
            Severity.OK,
            f"bookings run to {latest}",
            value=float(age),
        )

    def check_competitor_freshness(self, session: Session) -> CheckResult:
        """Competitor rates are still being collected.

        This is the check most likely to fire in practice: a scraper that gets
        blocked, or a producer that quietly died, leaves the pricing engine
        blind to the market while every other signal keeps working.
        """
        latest = session.execute(
            select(func.max(CompetitorPrice.collected_at))
        ).scalar_one_or_none()

        if latest is None:
            return CheckResult(
                "competitor_freshness",
                Severity.CRITICAL,
                "no competitor rates have ever been collected; the competitor "
                "adjustment will always be zero",
            )

        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
        hours = (datetime.now(timezone.utc) - latest).total_seconds() / 3600.0

        if hours > COMPETITOR_STALENESS_HOURS:
            return CheckResult(
                "competitor_freshness",
                Severity.WARNING,
                f"the newest competitor rate is {hours:.0f}h old; the market view "
                f"is stale and prices are drifting on our own signals alone",
                value=round(hours, 1),
                threshold=COMPETITOR_STALENESS_HOURS,
            )
        return CheckResult(
            "competitor_freshness",
            Severity.OK,
            f"newest competitor rate is {hours:.1f}h old",
            value=round(hours, 1),
        )

    @staticmethod
    def check_competitor_coverage(session: Session, *, horizon_days: int = 30) -> CheckResult:
        """Enough upcoming nights have a competitor rate to price against.

        The denominator is the *calendar* -- every night in the horizon -- not
        whatever happens to be in the feature store. An earlier version divided
        by the feature-store night count and reported 200% coverage, because the
        feature store stops at today while competitor rates run forward. A
        coverage ratio that can exceed 1.0 is not a coverage ratio.
        """
        today = date.today()
        horizon_end = today + timedelta(days=horizon_days)
        nights = horizon_days + 1

        covered = session.execute(
            select(func.count(func.distinct(CompetitorPrice.check_in_date))).where(
                CompetitorPrice.check_in_date >= today,
                CompetitorPrice.check_in_date <= horizon_end,
            )
        ).scalar_one()

        coverage = min(covered / nights, 1.0)
        if coverage < MIN_COMPETITOR_COVERAGE:
            return CheckResult(
                "competitor_coverage",
                Severity.WARNING,
                f"only {coverage:.0%} of the next {horizon_days} nights have a "
                f"competitor rate; the competitor features are mostly imputed and "
                f"the competitor adjustment is mostly zero",
                value=round(coverage, 4),
                threshold=MIN_COMPETITOR_COVERAGE,
            )
        return CheckResult(
            "competitor_coverage",
            Severity.OK,
            f"{coverage:.0%} of the next {horizon_days} nights have a competitor rate",
            value=round(coverage, 4),
        )

    @staticmethod
    def check_feature_freshness(session: Session) -> CheckResult:
        """The feature pipeline has run recently."""
        latest = session.execute(
            select(func.max(DemandFeature.computed_at))
        ).scalar_one_or_none()

        if latest is None:
            return CheckResult(
                "feature_freshness",
                Severity.CRITICAL,
                "the feature store has never been built; run scripts/build_features.py",
            )

        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
        hours = (datetime.now(timezone.utc) - latest).total_seconds() / 3600.0

        if hours > FEATURE_STALENESS_HOURS:
            return CheckResult(
                "feature_freshness",
                Severity.WARNING,
                f"features were last computed {hours:.0f}h ago; predictions are "
                f"being made against a stale picture",
                value=round(hours, 1),
                threshold=FEATURE_STALENESS_HOURS,
            )
        return CheckResult(
            "feature_freshness",
            Severity.OK,
            f"features computed {hours:.1f}h ago",
            value=round(hours, 1),
        )

    @staticmethod
    def check_feature_version(session: Session) -> CheckResult:
        """The stored features were built by the running code.

        A feature store full of v1 rows served to code that now produces v2
        features is train/serve skew that no accuracy metric will reveal.
        """
        versions = session.execute(
            select(DemandFeature.feature_version, func.count())
            .where(DemandFeature.feature_version.is_not(None))
            .group_by(DemandFeature.feature_version)
        ).all()

        if not versions:
            return CheckResult(
                "feature_version",
                Severity.CRITICAL,
                "no computed features found in the store",
            )

        stale = {v: n for v, n in versions if v != FEATURE_VERSION}
        if stale:
            return CheckResult(
                "feature_version",
                Severity.WARNING,
                f"the store holds features from other pipeline versions: {stale}. "
                f"The running code produces {FEATURE_VERSION}.",
                detail={"versions": dict(versions)},
            )
        return CheckResult(
            "feature_version",
            Severity.OK,
            f"all stored features are {FEATURE_VERSION}",
            detail={"versions": dict(versions)},
        )

    @staticmethod
    def check_feature_nulls(session: Session) -> CheckResult:
        """The columns a model needs are populated."""
        required = (
            DemandFeature.occupancy_rate,
            DemandFeature.competitor_rate,
            DemandFeature.search_demand,
            DemandFeature.historical_demand,
        )

        total = session.execute(
            select(func.count()).select_from(DemandFeature.__table__).where(
                DemandFeature.feature_version.is_not(None)
            )
        ).scalar_one()

        if not total:
            return CheckResult(
                "feature_nulls", Severity.CRITICAL, "no computed feature rows to check"
            )

        offenders: Dict[str, float] = {}
        for column in required:
            nulls = session.execute(
                select(func.count())
                .select_from(DemandFeature.__table__)
                .where(DemandFeature.feature_version.is_not(None), column.is_(None))
            ).scalar_one()
            if nulls:
                offenders[column.key] = round(nulls / total, 4)

        if offenders:
            worst = max(offenders.values())
            return CheckResult(
                "feature_nulls",
                Severity.CRITICAL if worst > 0.10 else Severity.WARNING,
                "null values in required feature column(s): "
                + ", ".join(f"{k} ({v:.1%})" for k, v in offenders.items()),
                value=worst,
                detail=offenders,
            )
        return CheckResult(
            "feature_nulls", Severity.OK, f"no nulls across {total:,} computed rows"
        )

    @staticmethod
    def check_target_range(session: Session) -> CheckResult:
        """Realised demand is inside a plausible range.

        Overbooking makes values above 1.0 legitimate, but a target of 3.0 means
        the inventory denominator is wrong -- and a model trained on it will
        price a full hotel as if it were three times full.
        """
        low, high, mean = session.execute(
            select(
                func.min(DemandFeature.target_demand),
                func.max(DemandFeature.target_demand),
                func.avg(DemandFeature.target_demand),
            ).where(DemandFeature.target_demand.is_not(None))
        ).one()

        if low is None:
            return CheckResult(
                "target_range", Severity.WARNING, "no labelled rows to check"
            )

        if high > 1.6 or low < 0:
            return CheckResult(
                "target_range",
                Severity.CRITICAL,
                f"realised demand spans {low:.2f} to {high:.2f}; values far above "
                f"1.0 indicate a wrong inventory denominator, not overbooking",
                value=float(high),
                threshold=1.6,
            )
        return CheckResult(
            "target_range",
            Severity.OK,
            f"realised demand spans {low:.2f} to {high:.2f}, mean {mean:.2f}",
            value=float(high),
        )

    @staticmethod
    def check_duplicate_grain(session: Session) -> CheckResult:
        """The feature store's grain is not violated.

        Enforced by a unique constraint, so this should never fire -- which is
        exactly why it is worth checking: if it ever does, the constraint was
        dropped by a migration and the models are training on duplicates.
        """
        duplicates = session.execute(
            select(func.count()).select_from(
                select(
                    DemandFeature.hotel_id,
                    DemandFeature.room_type,
                    DemandFeature.stay_date,
                )
                .group_by(
                    DemandFeature.hotel_id,
                    DemandFeature.room_type,
                    DemandFeature.stay_date,
                )
                .having(func.count() > 1)
                .subquery()
            )
        ).scalar_one()

        if duplicates:
            return CheckResult(
                "duplicate_grain",
                Severity.CRITICAL,
                f"{duplicates} duplicated (hotel, room, date) key(s) in the feature "
                f"store; the unique constraint is missing",
                value=float(duplicates),
            )
        return CheckResult("duplicate_grain", Severity.OK, "the feature store grain is intact")

    # -- orchestration --------------------------------------------------------- #

    def run(self, session: Session) -> DataQualityReport:
        """Run every check and return the report."""
        checks = [
            self.check_reference_data(session),
            self.check_booking_recency(session),
            self.check_competitor_freshness(session),
            self.check_competitor_coverage(session),
            self.check_feature_freshness(session),
            self.check_feature_version(session),
            self.check_feature_nulls(session),
            self.check_target_range(session),
            self.check_duplicate_grain(session),
        ]
        for check in checks:
            check.log()

        report = DataQualityReport(checks=checks)
        logger.info("Data quality: %s", report.summary())
        return report


__all__ = [
    "COMPETITOR_STALENESS_HOURS",
    "FEATURE_STALENESS_HOURS",
    "MIN_COMPETITOR_COVERAGE",
    "CheckResult",
    "DataMonitor",
    "DataQualityReport",
    "Severity",
]

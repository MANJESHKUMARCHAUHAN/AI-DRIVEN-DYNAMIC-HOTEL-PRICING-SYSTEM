"""The feature store: read raw tables, build features, persist them, read them back.

:mod:`features.feature_engineering` is pure computation and knows nothing about
databases. This module is the other half -- it owns the I/O, so the pipeline can
be tested on frames and used against PostgreSQL without either concern leaking
into the other.

``demand_features`` is written by two parties, and the split is enforced here:

* the streaming consumer owns the **exogenous** columns (search, weather,
  events, calendar) and upserts them as signals arrive;
* this module owns the **derived** columns and the target, and upserts only
  those -- so recomputing features never overwrites a signal, and a late-arriving
  signal never overwrites a computed feature.

It also owns ``feature_list.json``. Every model artifact is saved next to the
exact ordered feature list it was trained on, and :func:`validate_feature_list`
is called before inference. Train/serve skew is the failure that produces
confidently wrong prices with no error anywhere, so it is turned into a loud
exception at load time.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sqlalchemy import Table, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from database.models import Booking, CompetitorPrice, DemandFeature, Hotel, Room
from features.feature_engineering import (
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    KEY_COLUMNS,
    TARGET_COLUMN,
    FeatureBuilder,
    FeatureConfig,
)
from monitoring.logging_config import get_logger

logger = get_logger(__name__)

#: Rows per upsert batch.
CHUNK_SIZE = 2_000

#: Filename the ordered feature list is stored under, beside each artifact.
FEATURE_LIST_FILENAME = "feature_list.json"

#: Columns this module owns. Everything else in ``demand_features`` belongs to
#: the streaming consumer and is left untouched on update.
DERIVED_COLUMNS: Tuple[str, ...] = (
    "days_to_checkin",
    "total_rooms",
    "available_rooms",
    "occupancy_rate",
    "booking_count",
    "cancellation_count",
    "lead_time",
    "historical_demand",
    "current_room_price",
    "competitor_rate",
    "competitor_min_rate",
    "competitor_max_rate",
    "competitor_count",
    "target_demand",
    "feature_version",
    "computed_at",
)

#: Exogenous columns, supplied on INSERT so a brand-new row satisfies the NOT
#: NULL constraints, but never included in the UPDATE clause.
EXOGENOUS_COLUMNS: Tuple[str, ...] = (
    "day_of_week",
    "is_weekend",
    "season",
    "holiday_flag",
    "holiday_name",
    "local_event_score",
    "weather_score",
    "search_demand",
)


class FeatureVersionMismatch(RuntimeError):
    """A model is being served features it was not trained on."""


# --------------------------------------------------------------------------- #
# Feature list persistence
# --------------------------------------------------------------------------- #


def save_feature_list(
    directory: Path,
    *,
    columns: Sequence[str] = FEATURE_COLUMNS,
    version: str = FEATURE_VERSION,
    filename: str = FEATURE_LIST_FILENAME,
) -> Path:
    """Write the ordered feature list next to a model artifact."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(
        json.dumps(
            {
                "feature_version": version,
                "n_features": len(columns),
                "features": list(columns),
                "target": TARGET_COLUMN,
                "written_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote feature list (%d features) to %s", len(columns), path)
    return path


def load_feature_list(
    directory: Path, *, filename: str = FEATURE_LIST_FILENAME
) -> Dict[str, Any]:
    """Read a saved feature list."""
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Train a model first: python scripts/train_models.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def validate_feature_list(
    saved: Dict[str, Any], *, columns: Sequence[str] = FEATURE_COLUMNS
) -> None:
    """Assert that the running code produces what a model was trained on.

    Raises:
        FeatureVersionMismatch: On any difference in the feature set or its
            order. Order matters: a positional array handed to a model with two
            columns transposed produces plausible numbers and wrong prices.
    """
    expected = list(saved.get("features", []))
    actual = list(columns)

    if expected == actual:
        return

    missing = [c for c in expected if c not in actual]
    added = [c for c in actual if c not in expected]

    if missing or added:
        raise FeatureVersionMismatch(
            f"feature set changed since training. Missing now: {missing or 'none'}. "
            f"New: {added or 'none'}. Retrain, or serve a model trained on the "
            f"current feature set."
        )
    raise FeatureVersionMismatch(
        "feature set matches but the order differs; a positionally-indexed model "
        "would silently mis-read every row. Retrain."
    )


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


class FeatureStore:
    """Reads raw rows, builds features, and persists them."""

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self.config = config or FeatureConfig()
        self.builder = FeatureBuilder(self.config)

    # -- reading raw data --------------------------------------------------- #

    @staticmethod
    def load_raw(session: Session) -> Dict[str, pd.DataFrame]:
        """Pull the four raw frames the pipeline needs.

        Whole-table reads: at this scale (a few hundred thousand rows) that is
        seconds, and expressing the booking-curve arithmetic in SQL would trade
        a readable, testable pandas pipeline for an unreadable, untestable query.
        """
        def _frame(statement) -> pd.DataFrame:
            result = session.execute(statement)
            return pd.DataFrame(result.mappings().all())

        rooms = _frame(
            select(
                Room.hotel_id, Room.room_type, Room.room_count, Room.base_price
            )
        )
        hotels = _frame(select(Hotel.hotel_id, Hotel.city, Hotel.segment))
        bookings = _frame(
            select(
                Booking.hotel_id,
                Booking.room_type,
                Booking.booking_date,
                Booking.check_in_date,
                Booking.booking_count,
                Booking.cancellation_count,
                Booking.revenue,
                Booking.adr,
                Booking.lead_time_days,
            )
        )
        competitor = _frame(
            select(
                CompetitorPrice.hotel_id,
                CompetitorPrice.room_type,
                CompetitorPrice.competitor,
                CompetitorPrice.check_in_date,
                CompetitorPrice.price,
                CompetitorPrice.collected_at,
            )
        )
        signals = _frame(
            select(
                DemandFeature.hotel_id,
                DemandFeature.room_type,
                DemandFeature.stay_date,
                DemandFeature.search_demand,
                DemandFeature.weather_score,
                DemandFeature.local_event_score,
            )
        )

        logger.info(
            "Loaded raw data: %d booking, %d competitor, %d signal, %d room row(s)",
            len(bookings),
            len(competitor),
            len(signals),
            len(rooms),
        )
        return {
            "bookings": bookings,
            "competitor_prices": competitor,
            "signals": signals,
            "rooms": rooms,
            "hotels": hotels,
        }

    # -- building ----------------------------------------------------------- #

    def build(self, session: Session) -> pd.DataFrame:
        """Build the feature matrix from whatever is currently in the database."""
        raw = self.load_raw(session)
        if raw["bookings"].empty:
            raise ValueError(
                "the bookings table is empty; run scripts/seed_database.py first"
            )

        frame = self.builder.build(
            bookings=raw["bookings"],
            competitor_prices=raw["competitor_prices"],
            signals=raw["signals"],
            rooms=raw["rooms"],
        )

        # City drives the event calendar, so attach it for traceability even
        # though the score itself came from the signals.
        if not raw["hotels"].empty:
            frame = frame.merge(
                raw["hotels"][["hotel_id", "city"]], on="hotel_id", how="left"
            )
        return frame

    # -- persistence -------------------------------------------------------- #

    @staticmethod
    def _records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
        """Frame -> upsert payloads, with NaN converted to None."""
        columns = [
            *KEY_COLUMNS,
            *(c for c in EXOGENOUS_COLUMNS if c in frame.columns),
            *(c for c in DERIVED_COLUMNS if c in frame.columns),
        ]
        subset = frame[[c for c in dict.fromkeys(columns)]].copy()

        # ``stay_date`` arrives as a pandas Timestamp; the column is DATE.
        subset["stay_date"] = pd.to_datetime(subset["stay_date"]).dt.date

        for column in ("day_of_week", "total_rooms", "available_rooms",
                       "booking_count", "cancellation_count", "competitor_count",
                       "days_to_checkin"):
            if column in subset.columns:
                subset[column] = subset[column].round().astype("Int64")

        for column in ("is_weekend", "holiday_flag"):
            if column in subset.columns:
                subset[column] = subset[column].astype(bool)

        return subset.astype(object).where(pd.notna(subset), None).to_dict("records")

    def write(self, frame: pd.DataFrame, session: Session) -> int:
        """Upsert the computed features onto ``demand_features``.

        Only :data:`DERIVED_COLUMNS` appear in the UPDATE clause, so a rebuild
        cannot clobber a signal the streaming consumer wrote a moment earlier.

        Returns:
            Rows written.
        """
        records = self._records(frame)
        if not records:
            return 0

        table: Table = DemandFeature.__table__
        dialect = session.get_bind().dialect.name
        builder = pg_insert if dialect == "postgresql" else sqlite_insert

        written = 0
        for start in range(0, len(records), CHUNK_SIZE):
            chunk = records[start : start + CHUNK_SIZE]
            statement = builder(table).values(chunk)
            update = {
                column: statement.excluded[column]
                for column in DERIVED_COLUMNS
                if column in chunk[0]
            }
            statement = statement.on_conflict_do_update(
                index_elements=["hotel_id", "room_type", "stay_date"], set_=update
            )
            session.execute(statement)
            written += len(chunk)

        session.commit()
        logger.info("Upserted %d feature row(s) into demand_features", written)
        return written

    def build_and_store(self, session: Session) -> pd.DataFrame:
        """Build features and persist them in one step."""
        frame = self.build(session)
        self.write(frame, session)
        return frame

    # -- reading features back ---------------------------------------------- #

    @staticmethod
    def load_training_frame(
        session: Session,
        *,
        feature_version: Optional[str] = FEATURE_VERSION,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Read computed features back out, ready for training.

        Only rows with a target are returned: a row whose stay date has not
        happened yet has features but no label.
        """
        statement = select(DemandFeature).where(
            DemandFeature.target_demand.is_not(None)
        )
        if feature_version is not None:
            statement = statement.where(
                DemandFeature.feature_version == feature_version
            )
        if start_date is not None:
            statement = statement.where(DemandFeature.stay_date >= start_date)
        if end_date is not None:
            statement = statement.where(DemandFeature.stay_date <= end_date)

        rows = session.execute(statement).scalars().all()
        frame = pd.DataFrame([row.to_dict() for row in rows])
        logger.info("Loaded %d training row(s) from the feature store", len(frame))
        return frame

    @staticmethod
    def to_model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
        """Expand stored rows into the full model input.

        ``demand_features`` deliberately stores only the *base* columns. The
        season one-hots, the interaction terms and the ratios are pure functions
        of those, so persisting them would mean the same number lived in two
        places and could disagree.

        More importantly, this is what keeps train/serve parity real: the
        training path and :func:`~features.feature_engineering.build_serving_row`
        both derive those columns through
        :meth:`FeatureBuilder._add_derived_features`. One implementation, so a
        change to a ratio cannot land on one path and not the other.

        Args:
            frame: Rows as read from ``demand_features``.

        Returns:
            The same rows plus every column in
            :data:`~features.feature_engineering.FEATURE_COLUMNS`.
        """
        from database.models import RoomType, Season
        from features.calendars import holiday_proximity
        from features.feature_engineering import FEATURE_COLUMNS, FeatureBuilder

        if frame.empty:
            raise ValueError("cannot build a model matrix from an empty frame")

        matrix = frame.copy()
        matrix["stay_date"] = pd.to_datetime(matrix["stay_date"])
        matrix["room_type"] = matrix["room_type"].map(
            lambda v: v.value if isinstance(v, RoomType) else str(v)
        )

        # Booleans and integers arrive from SQLAlchemy in their own types; the
        # model wants a uniform float block.
        for column in ("is_weekend", "holiday_flag"):
            matrix[column] = matrix[column].astype(float)

        stay_dates = matrix["stay_date"].dt.date
        matrix["holiday_proximity"] = stay_dates.map(holiday_proximity).astype(float)

        seasons = matrix["season"].map(
            lambda v: v.value if isinstance(v, Season) else str(v)
        )
        for member in Season:
            matrix[f"season_{member.value}"] = (seasons == member.value).astype(float)

        # No competitor was visible at the snapshot; the rates were imputed.
        matrix["competitor_missing"] = (
            matrix["competitor_count"].fillna(0) == 0
        ).astype(float)

        matrix = FeatureBuilder._add_derived_features(matrix)

        missing = [c for c in FEATURE_COLUMNS if c not in matrix.columns]
        if missing:
            raise ValueError(
                f"stored rows cannot produce feature(s) {missing}; the feature "
                f"store schema and FEATURE_COLUMNS have diverged"
            )
        for column in FEATURE_COLUMNS:
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce")

        nulls = matrix[list(FEATURE_COLUMNS)].isna().sum()
        if nulls.any():
            raise ValueError(
                "stored rows produced null feature(s): "
                + ", ".join(nulls[nulls > 0].index)
            )
        return matrix

    @staticmethod
    def load_model_matrix(
        session: Session,
        *,
        feature_version: Optional[str] = FEATURE_VERSION,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Read the feature store and expand it into a model-ready matrix."""
        stored = FeatureStore.load_training_frame(
            session,
            feature_version=feature_version,
            start_date=start_date,
            end_date=end_date,
        )
        if stored.empty:
            raise ValueError(
                "the feature store holds no labelled rows for feature version "
                f"{feature_version!r}; run scripts/build_features.py first"
            )
        return FeatureStore.to_model_matrix(stored)

    @staticmethod
    def latest_for(
        session: Session, hotel_id: str, room_type: Any, stay_date: date
    ) -> Optional[Dict[str, Any]]:
        """The stored feature row for one night, or ``None``.

        Used at serving time to fill in whatever the API caller did not supply.
        """
        row = session.execute(
            select(DemandFeature).where(
                DemandFeature.hotel_id == hotel_id,
                DemandFeature.room_type == room_type,
                DemandFeature.stay_date == stay_date,
            )
        ).scalar_one_or_none()
        return row.to_dict() if row is not None else None

    @staticmethod
    def coverage(session: Session) -> Dict[str, Any]:
        """How much of the feature store is populated. For monitoring."""
        from sqlalchemy import func

        total, computed, labelled = session.execute(
            select(
                func.count(),
                func.count(DemandFeature.feature_version),
                func.count(DemandFeature.target_demand),
            ).select_from(DemandFeature.__table__)
        ).one()

        first, last = session.execute(
            select(func.min(DemandFeature.stay_date), func.max(DemandFeature.stay_date))
        ).one()

        return {
            "rows": int(total),
            "computed": int(computed),
            "labelled": int(labelled),
            "coverage": round(computed / total, 4) if total else 0.0,
            "first_stay_date": first,
            "last_stay_date": last,
        }


__all__ = [
    "CHUNK_SIZE",
    "DERIVED_COLUMNS",
    "EXOGENOUS_COLUMNS",
    "FEATURE_LIST_FILENAME",
    "FeatureStore",
    "FeatureVersionMismatch",
    "load_feature_list",
    "save_feature_list",
    "validate_feature_list",
]

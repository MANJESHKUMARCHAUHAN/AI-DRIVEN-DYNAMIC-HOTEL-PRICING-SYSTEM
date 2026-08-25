"""The feature pipeline: raw rows in, model-ready matrix out.

A feature row answers one question: **"what did we know about this night, at the
moment we had to price it?"** Every column is therefore computed as of a
*snapshot* taken ``days_to_checkin`` days before the stay, and nothing that
happened after that snapshot is allowed to reach the model.

Getting that wrong is the single most common way an ML pricing system produces
excellent offline metrics and useless online prices, so the leakage rules are
stated here rather than left implicit:

``occupancy_rate`` / ``available_rooms``
    Computed from **gross** rooms on the books at the snapshot -- every booking
    whose ``booking_date`` is on or before the snapshot. Cancellations are
    deliberately excluded: the schema records which *booking* a cancellation
    came from but not *when* it happened, so netting them off would use the
    future to describe the present.

``cancellation_count``
    Recent cancellation *pressure*, not this night's cancellations: the trailing
    28 days of cancellations on stay dates that had already completed by the
    snapshot. This is what a revenue manager actually has -- a cancellation
    forecast, never the actuals.

``historical_demand``
    Trailing 28-day mean realised demand for the same hotel and room type, over
    stay dates strictly on or before the snapshot date. The window end moves
    with the snapshot, so a 30-day-out row sees a month less history than a
    same-day row -- exactly as it would in production.

``competitor_*``
    Only observations collected on or before the snapshot, and the most recent
    one per competitor. A long horizon can legitimately find nothing at all,
    which is why ``competitor_missing`` is a feature rather than an exception.

``target_demand``
    Final **net** rooms sold divided by inventory, known only after the stay.
    Never a feature.

**Snapshot horizons.** One row per ``(hotel, room type, stay date)``, with the
horizon drawn deterministically from the row's own key and weighted towards the
lead times where pricing decisions are actually made. The alternative -- a full
panel of every stay date at every horizon -- yields more rows but heavily
correlated ones, and would change the feature store's grain. The horizon is
itself a feature, so the model still learns how the picture differs at 30 days
out versus same-day.

The whole module is deterministic and side-effect free: same input, same output,
no clock reads, no network. That is what makes train/serve parity testable, and
:func:`build_serving_row` shares its column construction with the training path
so the two cannot drift apart.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from database.models import RoomType, Season
from features.calendars import (
    event_score,
    holiday_name,
    holiday_proximity,
    is_holiday,
    is_weekend,
    season_of,
)
from monitoring.logging_config import get_logger

logger = get_logger(__name__)

#: Bumped whenever a feature's definition changes. Persisted with every model
#: and checked at load time, so a model trained on v1 can never be served v2
#: features -- the failure mode that produces confidently wrong prices.
FEATURE_VERSION = "v1"

#: Snapshot horizons and their sampling weights. Concentrated inside three
#: weeks because that is where the booking curve is steep and where a price
#: change still changes the outcome; the long tail keeps the model honest about
#: dates where no competitor data exists yet.
HORIZON_CHOICES: Tuple[int, ...] = (0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60)
HORIZON_WEIGHTS: Tuple[float, ...] = (
    0.06, 0.09, 0.09, 0.09, 0.10, 0.11, 0.10, 0.10, 0.09, 0.08, 0.05, 0.04
)

#: Days of pickup summarised by ``booking_count``.
PICKUP_WINDOW_DAYS = 7

#: Days of history behind ``historical_demand`` and ``cancellation_count``.
HISTORY_WINDOW_DAYS = 28

#: Occupancy above this is overbooking. Kept rather than clipped to 1.0 in the
#: target, because "we sold 108% of the rooms" is real information about demand.
MAX_TARGET_DEMAND = 1.5


# --------------------------------------------------------------------------- #
# The feature contract
# --------------------------------------------------------------------------- #

#: The features named in the specification, in a fixed order.
BASE_FEATURES: Tuple[str, ...] = (
    "occupancy_rate",
    "available_rooms",
    "total_rooms",
    "booking_count",
    "cancellation_count",
    "competitor_rate",
    "competitor_min_rate",
    "competitor_max_rate",
    "competitor_count",
    "days_to_checkin",
    "lead_time",
    "search_demand",
    "historical_demand",
    "current_room_price",
    "is_weekend",
    "day_of_week",
    "holiday_flag",
    "local_event_score",
    "weather_score",
)

#: Season is categorical with no ordering, so it is one-hot encoded rather than
#: given an arbitrary integer code a tree would happily split on as if ordered.
SEASON_FEATURES: Tuple[str, ...] = tuple(f"season_{s.value}" for s in Season)

#: Derived features. Each earns its place by expressing something the raw
#: columns only imply.
DERIVED_FEATURES: Tuple[str, ...] = (
    # Where we sit against the market. The ratio matters more than either level.
    "price_to_competitor",
    # How dispersed the competitive set is. A wide spread means weak price
    # discipline and more room to move.
    "competitor_spread",
    # Rooms per day being taken right now -- the slope of the booking curve.
    "pickup_velocity",
    # The interaction from docs/architecture.md §10: 80% full at 30 days out is
    # a very different situation from 80% full on the day.
    "occupancy_x_lead",
    # Forward-looking interest, amplified by whatever is happening in the city.
    "demand_pressure",
    # Explicit missingness flag. Trees can then learn "when the market is
    # invisible, fall back on our own signals" instead of trusting an imputation.
    "competitor_missing",
    # Holiday pressure including the days either side, not just the day itself.
    "holiday_proximity",
)

#: The complete, ordered feature vector. This tuple *is* the contract between
#: training and serving; it is written into every model artifact.
FEATURE_COLUMNS: Tuple[str, ...] = BASE_FEATURES + SEASON_FEATURES + DERIVED_FEATURES

#: The supervised target. Never a feature.
TARGET_COLUMN = "target_demand"

#: Identify a row without being part of the model input.
KEY_COLUMNS: Tuple[str, ...] = ("hotel_id", "room_type", "stay_date")


@dataclass
class FeatureConfig:
    """Tunables for the pipeline. Defaults match :data:`FEATURE_VERSION`."""

    horizon_choices: Tuple[int, ...] = HORIZON_CHOICES
    horizon_weights: Tuple[float, ...] = HORIZON_WEIGHTS
    pickup_window_days: int = PICKUP_WINDOW_DAYS
    history_window_days: int = HISTORY_WINDOW_DAYS
    feature_version: str = FEATURE_VERSION
    #: Salt for horizon selection. Changing it reshuffles horizons without
    #: touching any other part of the pipeline.
    horizon_seed: int = 20260824
    #: Rows whose stay date is within this many days of the end of the data are
    #: dropped: their booking curve is still filling, so their target is not a
    #: final number.
    incomplete_tail_days: int = 1

    def __post_init__(self) -> None:
        if len(self.horizon_choices) != len(self.horizon_weights):
            raise ValueError("horizon_choices and horizon_weights must be the same length")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def assign_horizon(
    hotel_id: str, room_type: str, stay_date: date, config: FeatureConfig
) -> int:
    """Pick this row's snapshot horizon, deterministically from its own key.

    A hash rather than an RNG draw so the choice is reproducible without
    threading generator state through the pipeline, and stable if rows are
    reordered or the dataset is regenerated with more hotels.
    """
    digest = hashlib.blake2b(
        f"{config.horizon_seed}|{hotel_id}|{room_type}|{stay_date.isoformat()}".encode(),
        digest_size=8,
    ).digest()
    draw = int.from_bytes(digest, "big") / float(1 << 64)

    cumulative = 0.0
    total = sum(config.horizon_weights)
    for horizon, weight in zip(config.horizon_choices, config.horizon_weights):
        cumulative += weight / total
        if draw < cumulative:
            return horizon
    return config.horizon_choices[-1]


def _reverse_cumsum(
    frame: pd.DataFrame, keys: Sequence[str], columns: Sequence[str]
) -> pd.DataFrame:
    """Cumulative sums taken from the *end* of each group.

    Bookings are stored one row per lead time. Reversing the cumulative sum
    turns "bookings taken at exactly this lead" into "bookings on the books at
    this lead or earlier" -- which is on-the-books occupancy, computed in one
    vectorised pass instead of a per-row query.
    """
    flipped = frame.iloc[::-1]
    sums = flipped.groupby(list(keys), sort=False)[list(columns)].cumsum()
    return sums.iloc[::-1]


def _otb_at_horizon(
    targets: pd.DataFrame, curves: pd.DataFrame, horizon_column: str
) -> pd.DataFrame:
    """On-the-books totals for each target row at its own horizon.

    ``merge_asof(direction="forward")`` finds, per group, the first booking row
    whose lead time is at or beyond the horizon -- i.e. the earliest booking
    still in the past at snapshot time. Rows with no such booking (nothing was
    on the books yet) come back as NaN and are filled with zero by the caller.

    The row order is restored through an explicit ``_row`` column rather than
    ``sort_index()``. ``merge_asof`` requires its inputs sorted by the join key
    and returns a *fresh* ``RangeIndex``, so the original index is destroyed --
    and ``sort_index()`` on the result silently re-sorts by position, leaving
    every row holding some other row's on-the-books total. That misalignment is
    invisible in aggregate (the column still has a sensible distribution) and
    catastrophic in fact.
    """
    left = targets.reset_index(drop=True).rename_axis("_row").reset_index()
    left = left.sort_values(horizon_column, kind="mergesort")
    right = curves.sort_values("lead_time_days", kind="mergesort")

    merged = pd.merge_asof(
        left,
        right,
        left_on=horizon_column,
        right_on="lead_time_days",
        by=["hotel_id", "room_type", "stay_date"],
        direction="forward",
    )
    return merged.set_index("_row").sort_index()


def _rolling_history(
    daily: pd.DataFrame, column: str, window: int, how: str
) -> pd.DataFrame:
    """Trailing window statistic per hotel and room type, on a complete calendar.

    Reindexed to every date in the range first: a rolling window over rows would
    silently treat a missing day as no day at all, making "28 days" mean
    something different for every group.
    """
    frames: List[pd.DataFrame] = []
    for (hotel_id, room_type), group in daily.groupby(["hotel_id", "room_type"]):
        series = (
            group.set_index("stay_date")[column]
            .sort_index()
            .asfreq("D")
        )
        rolled = getattr(series.rolling(window, min_periods=1), how)()
        frames.append(
            pd.DataFrame(
                {
                    "hotel_id": hotel_id,
                    "room_type": room_type,
                    "as_of_date": rolled.index,
                    f"{column}_history": rolled.to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #


class FeatureBuilder:
    """Turns the raw tables into a training matrix.

    Example::

        builder = FeatureBuilder()
        matrix = builder.build(bookings, competitor_prices, signals, rooms)
        X, y = builder.split(matrix)
    """

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self.config = config or FeatureConfig()

    # -- public ------------------------------------------------------------ #

    def build(
        self,
        bookings: pd.DataFrame,
        competitor_prices: pd.DataFrame,
        signals: pd.DataFrame,
        rooms: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build the full feature matrix.

        Args:
            bookings: Pickup rows -- one per (hotel, room, booking date, stay).
            competitor_prices: Observed competitor rates with ``collected_at``.
            signals: Exogenous per-night signals (search, weather, events).
            rooms: Inventory and base price per hotel and room type.

        Returns:
            One row per (hotel, room type, stay date) with
            :data:`FEATURE_COLUMNS`, :data:`TARGET_COLUMN` and the key columns.
        """
        bookings = self._normalise_bookings(bookings)
        rooms = self._normalise_rooms(rooms)

        frame = self._skeleton(bookings, rooms)
        frame = self._add_booking_curve_features(frame, bookings)
        frame = self._add_history_features(frame, bookings)
        frame = self._add_competitor_features(frame, competitor_prices)
        frame = self._add_signal_features(frame, signals)
        frame = self._add_calendar_features(frame)
        frame = self._add_derived_features(frame)
        frame = self._finalise(frame)

        logger.info(
            "Built %d feature rows (%d columns, version %s)",
            len(frame),
            len(FEATURE_COLUMNS),
            self.config.feature_version,
        )
        return frame

    @staticmethod
    def split(matrix: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate the model input from the target."""
        return matrix[list(FEATURE_COLUMNS)], matrix[TARGET_COLUMN]

    # -- normalisation ----------------------------------------------------- #

    @staticmethod
    def _normalise_bookings(bookings: pd.DataFrame) -> pd.DataFrame:
        """Coerce dtypes and drop rows that cannot contribute.

        Defensive rather than trusting: this frame may have come from a CSV, a
        SQL result or a test fixture, and each renders dates differently.
        """
        frame = bookings.copy()
        for column in ("booking_date", "check_in_date"):
            frame[column] = pd.to_datetime(frame[column]).dt.normalize()

        frame["room_type"] = frame["room_type"].map(
            lambda v: v.value if isinstance(v, RoomType) else str(v)
        )
        frame = frame.rename(columns={"check_in_date": "stay_date"})

        for column in ("booking_count", "cancellation_count", "revenue", "adr"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

        # int64, to match the horizon column merge_asof joins it against.
        frame["lead_time_days"] = (
            pd.to_numeric(frame["lead_time_days"], errors="coerce")
            .fillna(0)
            .round()
            .astype("int64")
        )

        # A pickup row with no rooms in it carries no information and would only
        # add zero-weight rows to every weighted average downstream.
        return frame[frame["booking_count"] > 0].reset_index(drop=True)

    @staticmethod
    def _normalise_rooms(rooms: pd.DataFrame) -> pd.DataFrame:
        frame = rooms.copy()
        frame["room_type"] = frame["room_type"].map(
            lambda v: v.value if isinstance(v, RoomType) else str(v)
        )
        frame["room_count"] = pd.to_numeric(frame["room_count"]).astype(int)
        frame["base_price"] = pd.to_numeric(frame["base_price"]).astype(float)
        return frame[["hotel_id", "room_type", "room_count", "base_price"]]

    # -- skeleton and target ------------------------------------------------ #

    def _skeleton(self, bookings: pd.DataFrame, rooms: pd.DataFrame) -> pd.DataFrame:
        """One row per (hotel, room type, stay date), with the target attached.

        The target is final net rooms sold over inventory -- the only place in
        the pipeline where post-stay information is allowed.
        """
        totals = (
            bookings.groupby(["hotel_id", "room_type", "stay_date"], as_index=False)
            .agg(
                gross_final=("booking_count", "sum"),
                cancels_final=("cancellation_count", "sum"),
            )
        )
        totals["net_sold"] = totals["gross_final"] - totals["cancels_final"]

        frame = totals.merge(rooms, on=["hotel_id", "room_type"], how="inner")
        frame[TARGET_COLUMN] = (
            frame["net_sold"] / frame["room_count"]
        ).clip(0.0, MAX_TARGET_DEMAND)

        # Nights at the very end of the window are still taking bookings, so
        # their "final" total is not final.
        cutoff = frame["stay_date"].max() - pd.Timedelta(
            days=self.config.incomplete_tail_days
        )
        dropped = int((frame["stay_date"] > cutoff).sum())
        if dropped:
            logger.info(
                "Dropped %d row(s) whose booking curve is still open (stay date > %s)",
                dropped,
                cutoff.date(),
            )
        frame = frame[frame["stay_date"] <= cutoff].reset_index(drop=True)

        if frame.empty:
            raise ValueError(
                "no stay dates survived the incomplete-tail filter. Every night "
                f"in the input falls after {cutoff.date()}, so none has a final "
                "booking total. Supply a longer history, or set "
                "FeatureConfig(incomplete_tail_days=0) if the data is known to "
                "be complete."
            )

        # Typed explicitly rather than inferred: an empty or all-integer list
        # infers differently, and merge_asof refuses to join an int64 key to a
        # float64 one with an error that names neither column.
        frame["days_to_checkin"] = pd.Series(
            [
                assign_horizon(h, r, s.date(), self.config)
                for h, r, s in zip(
                    frame["hotel_id"], frame["room_type"], frame["stay_date"]
                )
            ],
            index=frame.index,
            dtype="int64",
        )
        frame["snapshot_date"] = frame["stay_date"] - pd.to_timedelta(
            frame["days_to_checkin"], unit="D"
        )
        return frame

    # -- booking curve ------------------------------------------------------ #

    def _add_booking_curve_features(
        self, frame: pd.DataFrame, bookings: pd.DataFrame
    ) -> pd.DataFrame:
        """On-the-books occupancy, pickup, lead time and achieved rate.

        All four come from suffix sums over the lead-time axis, evaluated at
        each row's own horizon. Nothing here uses a cancellation.
        """
        curves = bookings.sort_values(
            ["hotel_id", "room_type", "stay_date", "lead_time_days"], kind="mergesort"
        ).reset_index(drop=True)

        curves["lead_weighted"] = curves["lead_time_days"] * curves["booking_count"]
        curves["rate_weighted"] = curves["adr"] * curves["booking_count"]

        sums = _reverse_cumsum(
            curves,
            keys=("hotel_id", "room_type", "stay_date"),
            columns=("booking_count", "lead_weighted", "rate_weighted"),
        )
        curves["otb_rooms"] = sums["booking_count"]
        curves["otb_lead"] = sums["lead_weighted"]
        curves["otb_rate"] = sums["rate_weighted"]

        lookup = curves[
            ["hotel_id", "room_type", "stay_date", "lead_time_days",
             "otb_rooms", "otb_lead", "otb_rate"]
        ]

        at_horizon = _otb_at_horizon(frame, lookup, "days_to_checkin")

        # The same lookup one pickup-window earlier: the difference is how many
        # rooms were taken during the window.
        frame["_window_start"] = frame["days_to_checkin"] + self.config.pickup_window_days
        before_window = _otb_at_horizon(
            frame[["hotel_id", "room_type", "stay_date", "_window_start"]],
            lookup[["hotel_id", "room_type", "stay_date", "lead_time_days", "otb_rooms"]],
            "_window_start",
        )

        otb = at_horizon["otb_rooms"].fillna(0.0).to_numpy()
        otb_lead = at_horizon["otb_lead"].fillna(0.0).to_numpy()
        otb_rate = at_horizon["otb_rate"].fillna(0.0).to_numpy()
        otb_before = before_window["otb_rooms"].fillna(0.0).to_numpy()

        inventory = frame["room_count"].to_numpy(dtype=float)

        frame["total_rooms"] = inventory
        frame["occupancy_rate"] = np.clip(otb / inventory, 0.0, 1.0)
        frame["available_rooms"] = np.maximum(inventory - otb, 0.0)
        frame["booking_count"] = np.maximum(otb - otb_before, 0.0)

        # Weighted means, guarding the empty-book case where the denominator is
        # zero: fall back to the horizon itself and to the room's base price.
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_lead = np.where(otb > 0, otb_lead / np.maximum(otb, 1e-9), np.nan)
            mean_rate = np.where(otb > 0, otb_rate / np.maximum(otb, 1e-9), np.nan)

        frame["lead_time"] = np.where(
            np.isnan(mean_lead), frame["days_to_checkin"].to_numpy(dtype=float), mean_lead
        )
        frame["current_room_price"] = np.where(
            np.isnan(mean_rate), frame["base_price"].to_numpy(dtype=float), mean_rate
        )
        return frame.drop(columns=["_window_start"])

    # -- history ------------------------------------------------------------ #

    def _add_history_features(
        self, frame: pd.DataFrame, bookings: pd.DataFrame
    ) -> pd.DataFrame:
        """Trailing demand and cancellation pressure, as of each snapshot.

        The window ends at the snapshot date, not at the stay date, so a row
        priced 30 days out genuinely sees a month less history.
        """
        daily = (
            bookings.groupby(["hotel_id", "room_type", "stay_date"], as_index=False)
            .agg(
                gross=("booking_count", "sum"),
                cancels=("cancellation_count", "sum"),
            )
        )
        inventory = frame[["hotel_id", "room_type", "room_count"]].drop_duplicates()
        daily = daily.merge(inventory, on=["hotel_id", "room_type"], how="left")
        daily["realised_demand"] = (
            (daily["gross"] - daily["cancels"]) / daily["room_count"]
        ).clip(0.0, MAX_TARGET_DEMAND)

        demand_history = _rolling_history(
            daily, "realised_demand", self.config.history_window_days, "mean"
        )
        cancel_history = _rolling_history(
            daily, "cancels", self.config.history_window_days, "sum"
        )

        frame = frame.merge(
            demand_history,
            left_on=["hotel_id", "room_type", "snapshot_date"],
            right_on=["hotel_id", "room_type", "as_of_date"],
            how="left",
        ).drop(columns=["as_of_date"])
        frame = frame.merge(
            cancel_history,
            left_on=["hotel_id", "room_type", "snapshot_date"],
            right_on=["hotel_id", "room_type", "as_of_date"],
            how="left",
        ).drop(columns=["as_of_date"])

        frame = frame.rename(
            columns={
                "realised_demand_history": "historical_demand",
                "cancels_history": "cancellation_count",
            }
        )

        # Rows near the start of the window have no history behind them. Filling
        # with the group's own mean, rather than a global constant, keeps a
        # quiet property from inheriting a busy one's baseline.
        group_mean = frame.groupby(["hotel_id", "room_type"])["historical_demand"]
        frame["historical_demand"] = frame["historical_demand"].fillna(
            group_mean.transform("mean")
        ).fillna(0.0)
        frame["cancellation_count"] = frame["cancellation_count"].fillna(0.0)
        return frame

    # -- competitors -------------------------------------------------------- #

    def _add_competitor_features(
        self, frame: pd.DataFrame, competitor_prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Latest competitor rate per source, as of the snapshot.

        Observations collected after the snapshot are invisible. A horizon
        beyond the furthest observation legitimately yields nothing, and
        ``competitor_missing`` records that rather than hiding it.
        """
        if competitor_prices.empty:
            return self._empty_competitor_columns(frame)

        rates = competitor_prices.copy()
        rates["room_type"] = rates["room_type"].map(
            lambda v: v.value if isinstance(v, RoomType) else str(v)
        )
        rates["stay_date"] = pd.to_datetime(rates["check_in_date"]).dt.normalize()
        rates["observed_on"] = (
            pd.to_datetime(rates["collected_at"], utc=True).dt.tz_localize(None).dt.normalize()
        )
        rates["observation_lead"] = (
            rates["stay_date"] - rates["observed_on"]
        ).dt.days
        rates["price"] = pd.to_numeric(rates["price"], errors="coerce")
        rates = rates.dropna(subset=["price"])

        keys = ["hotel_id", "room_type", "stay_date"]
        joined = frame[keys + ["days_to_checkin"]].merge(rates, on=keys, how="inner")

        # Only what had been collected by the snapshot.
        visible = joined[joined["observation_lead"] >= joined["days_to_checkin"]]

        if visible.empty:
            return self._empty_competitor_columns(frame)

        # Most recent observation per competitor: the smallest still-valid lead.
        freshest = (
            visible.sort_values("observation_lead", kind="mergesort")
            .groupby(keys + ["competitor"], as_index=False)
            .first()
        )

        aggregated = freshest.groupby(keys, as_index=False).agg(
            competitor_rate=("price", "mean"),
            competitor_min_rate=("price", "min"),
            competitor_max_rate=("price", "max"),
            competitor_count=("price", "size"),
        )

        frame = frame.merge(aggregated, on=keys, how="left")
        frame["competitor_missing"] = frame["competitor_rate"].isna().astype(float)
        frame["competitor_count"] = frame["competitor_count"].fillna(0.0)

        # Impute a missing market with our own base price: the neutral
        # assumption is "the market is where we are", which makes the competitor
        # adjustment zero rather than inventing a direction.
        fallback = frame["base_price"]
        for column in ("competitor_rate", "competitor_min_rate", "competitor_max_rate"):
            frame[column] = frame[column].fillna(fallback)

        missing = int(frame["competitor_missing"].sum())
        if missing:
            logger.info(
                "%d of %d row(s) (%.1f%%) had no competitor data at their snapshot",
                missing,
                len(frame),
                100.0 * missing / max(len(frame), 1),
            )
        return frame

    @staticmethod
    def _empty_competitor_columns(frame: pd.DataFrame) -> pd.DataFrame:
        """Fall back to base price everywhere when no rates are visible at all."""
        logger.warning(
            "No competitor observations were visible for any row; the competitor "
            "features fall back entirely to base price"
        )
        frame["competitor_rate"] = frame["base_price"]
        frame["competitor_min_rate"] = frame["base_price"]
        frame["competitor_max_rate"] = frame["base_price"]
        frame["competitor_count"] = 0.0
        frame["competitor_missing"] = 1.0
        return frame

    # -- exogenous signals --------------------------------------------------- #

    def _add_signal_features(
        self, frame: pd.DataFrame, signals: pd.DataFrame
    ) -> pd.DataFrame:
        """Search interest, weather and event scores.

        Exogenous: facts about the world rather than about our hotel, and
        therefore knowable for dates that have not happened yet.
        """
        defaults = {"search_demand": 0.0, "weather_score": 0.5, "local_event_score": 0.0}

        if signals is None or signals.empty:
            for column, value in defaults.items():
                frame[column] = value
            return frame

        wanted = ["hotel_id", "room_type", "stay_date", *defaults]
        available = [c for c in wanted if c in signals.columns]
        prepared = signals[available].copy()
        prepared["room_type"] = prepared["room_type"].map(
            lambda v: v.value if isinstance(v, RoomType) else str(v)
        )
        prepared["stay_date"] = pd.to_datetime(prepared["stay_date"]).dt.normalize()
        prepared = prepared.drop_duplicates(subset=["hotel_id", "room_type", "stay_date"])

        frame = frame.merge(
            prepared, on=["hotel_id", "room_type", "stay_date"], how="left"
        )
        for column, value in defaults.items():
            if column not in frame.columns:
                frame[column] = value
            else:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(value)
        return frame

    # -- calendar ------------------------------------------------------------ #

    def _add_calendar_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Day of week, weekend, season and holiday flags.

        Recomputed from :mod:`features.calendars` rather than read from the
        database, so training and serving share one definition even if a
        producer disagrees about which day is a holiday.
        """
        stay_dates = frame["stay_date"].dt.date

        frame["day_of_week"] = frame["stay_date"].dt.dayofweek.astype(float)
        frame["is_weekend"] = stay_dates.map(is_weekend).astype(float)
        frame["holiday_flag"] = stay_dates.map(is_holiday).astype(float)
        frame["holiday_proximity"] = stay_dates.map(holiday_proximity).astype(float)
        frame["holiday_name"] = stay_dates.map(holiday_name)

        seasons = stay_dates.map(lambda d: season_of(d).value)
        frame["season"] = seasons
        for season in Season:
            frame[f"season_{season.value}"] = (seasons == season.value).astype(float)

        # The event score needs the city, which lives on the hotel. When the
        # caller did not join it, fall back to whatever the signals carried.
        if "city" in frame.columns:
            frame["local_event_score"] = [
                event_score(city, stay) if isinstance(city, str) else fallback
                for city, stay, fallback in zip(
                    frame["city"], stay_dates, frame["local_event_score"]
                )
            ]
        return frame

    # -- derived -------------------------------------------------------------- #

    @staticmethod
    def _add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
        """Interactions and ratios that the raw columns only imply."""
        competitor = frame["competitor_rate"].replace(0.0, np.nan)

        frame["price_to_competitor"] = (
            frame["current_room_price"] / competitor
        ).fillna(1.0).clip(0.2, 5.0)

        frame["competitor_spread"] = (
            (frame["competitor_max_rate"] - frame["competitor_min_rate"]) / competitor
        ).fillna(0.0).clip(0.0, 2.0)

        frame["pickup_velocity"] = frame["booking_count"] / float(PICKUP_WINDOW_DAYS)

        # Normalised so the interaction stays on a comparable scale to the
        # occupancy term itself: 1.0 is "full, and checking in today".
        lead_factor = 1.0 - (frame["days_to_checkin"] / float(max(HORIZON_CHOICES)))
        frame["occupancy_x_lead"] = frame["occupancy_rate"] * lead_factor

        frame["demand_pressure"] = frame["search_demand"] * (
            1.0 + frame["local_event_score"]
        )
        return frame

    # -- finalisation --------------------------------------------------------- #

    def _finalise(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Enforce the contract: every feature present, finite and typed.

        A NaN or an infinity reaching the model is a silent failure -- some
        estimators accept them and produce nonsense -- so the check is explicit
        and loud.
        """
        missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"feature pipeline produced no {missing} column(s)")

        for column in FEATURE_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        infinite = np.isinf(frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float))
        if infinite.any():
            offenders = [
                FEATURE_COLUMNS[i] for i in np.unique(np.argwhere(infinite)[:, 1])
            ]
            raise ValueError(f"non-finite values in feature(s): {offenders}")

        nulls = frame[list(FEATURE_COLUMNS)].isna().sum()
        if nulls.any():
            raise ValueError(
                "null values survived the pipeline in: "
                + ", ".join(nulls[nulls > 0].index)
            )

        frame["feature_version"] = self.config.feature_version
        frame["computed_at"] = datetime.now(timezone.utc)

        ordered = (
            list(KEY_COLUMNS)
            + ["snapshot_date", "feature_version", "computed_at",
               "season", "holiday_name", "base_price", "net_sold"]
            + list(FEATURE_COLUMNS)
            + [TARGET_COLUMN]
        )
        return frame[[c for c in ordered if c in frame.columns]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Serving
# --------------------------------------------------------------------------- #


def build_serving_row(
    *,
    hotel_id: str,
    room_type: RoomType,
    check_in_date: date,
    base_price: float,
    total_rooms: int,
    city: Optional[str] = None,
    as_of: Optional[date] = None,
    occupancy_rate: Optional[float] = None,
    available_rooms: Optional[int] = None,
    current_price: Optional[float] = None,
    competitor_rate: Optional[float] = None,
    competitor_min_rate: Optional[float] = None,
    competitor_max_rate: Optional[float] = None,
    competitor_count: Optional[int] = None,
    booking_count: Optional[float] = None,
    cancellation_count: Optional[float] = None,
    lead_time: Optional[float] = None,
    search_demand: Optional[float] = None,
    weather_score: Optional[float] = None,
    local_event_score: Optional[float] = None,
    historical_demand: Optional[float] = None,
) -> pd.DataFrame:
    """Build a single, model-ready feature row for inference.

    This is the train/serve parity seam. It produces exactly
    :data:`FEATURE_COLUMNS`, in exactly that order, using the same calendar
    module and the same derived-feature arithmetic as the training path -- so a
    column can only be added in one place and forgotten in the other if
    ``FEATURE_COLUMNS`` itself is edited, which the tests check.

    Every optional argument is a value the caller may know from its own request;
    anything omitted falls back to the neutral assumption, which for the market
    is "the competition is where we are".

    Args:
        base_price: The room's rack rate, the fallback for every price feature.
        total_rooms: Inventory for this room type.
        as_of: Pricing date, for the ``days_to_checkin`` horizon. Defaults to
            today (UTC).

    Returns:
        A one-row frame containing :data:`FEATURE_COLUMNS`.
    """
    as_of = as_of or datetime.now(timezone.utc).date()
    horizon = max((check_in_date - as_of).days, 0)

    if occupancy_rate is None:
        occupancy_rate = (
            1.0 - (available_rooms / total_rooms)
            if available_rooms is not None and total_rooms
            else 0.0
        )
    occupancy_rate = float(np.clip(occupancy_rate, 0.0, 1.0))

    if available_rooms is None:
        available_rooms = int(round(total_rooms * (1.0 - occupancy_rate)))

    competitor_missing = competitor_rate is None
    competitor_rate = float(competitor_rate if competitor_rate is not None else base_price)
    competitor_min_rate = float(
        competitor_min_rate if competitor_min_rate is not None else competitor_rate
    )
    competitor_max_rate = float(
        competitor_max_rate if competitor_max_rate is not None else competitor_rate
    )

    current_price = float(current_price if current_price is not None else base_price)
    season = season_of(check_in_date)

    row: Dict[str, Any] = {
        "occupancy_rate": occupancy_rate,
        "available_rooms": float(available_rooms),
        "total_rooms": float(total_rooms),
        "booking_count": float(booking_count or 0.0),
        "cancellation_count": float(cancellation_count or 0.0),
        "competitor_rate": competitor_rate,
        "competitor_min_rate": competitor_min_rate,
        "competitor_max_rate": competitor_max_rate,
        "competitor_count": float(
            competitor_count if competitor_count is not None else (0 if competitor_missing else 1)
        ),
        "days_to_checkin": float(horizon),
        "lead_time": float(lead_time if lead_time is not None else horizon),
        "search_demand": float(search_demand if search_demand is not None else 0.0),
        "historical_demand": float(
            historical_demand if historical_demand is not None else occupancy_rate
        ),
        "current_room_price": current_price,
        "is_weekend": float(is_weekend(check_in_date)),
        "day_of_week": float(check_in_date.weekday()),
        "holiday_flag": float(is_holiday(check_in_date)),
        "local_event_score": float(
            local_event_score
            if local_event_score is not None
            else (event_score(city, check_in_date) if city else 0.0)
        ),
        "weather_score": float(weather_score if weather_score is not None else 0.5),
        "holiday_proximity": float(holiday_proximity(check_in_date)),
        "competitor_missing": float(competitor_missing),
    }
    for member in Season:
        row[f"season_{member.value}"] = float(member is season)

    frame = pd.DataFrame([row])
    frame["base_price"] = base_price
    frame = FeatureBuilder._add_derived_features(frame)

    return frame[list(FEATURE_COLUMNS)]


def describe_features() -> pd.DataFrame:
    """The feature contract as a table, for the docs and the dashboard."""
    groups = (
        [(c, "inventory / pickup") for c in BASE_FEATURES[:5]]
        + [(c, "competitive set") for c in BASE_FEATURES[5:9]]
        + [(c, "timing") for c in BASE_FEATURES[9:11]]
        + [(c, "demand signal") for c in BASE_FEATURES[11:14]]
        + [(c, "calendar") for c in BASE_FEATURES[14:19]]
        + [(c, "calendar") for c in SEASON_FEATURES]
        + [(c, "derived") for c in DERIVED_FEATURES]
    )
    return pd.DataFrame(groups, columns=["feature", "group"])


__all__ = [
    "BASE_FEATURES",
    "DERIVED_FEATURES",
    "FEATURE_COLUMNS",
    "FEATURE_VERSION",
    "HORIZON_CHOICES",
    "KEY_COLUMNS",
    "SEASON_FEATURES",
    "TARGET_COLUMN",
    "FeatureBuilder",
    "FeatureConfig",
    "assign_horizon",
    "build_serving_row",
    "describe_features",
]

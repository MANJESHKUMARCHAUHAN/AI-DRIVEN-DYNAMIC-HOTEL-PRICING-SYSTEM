"""Tests for the feature pipeline.

The leakage tests are the point of this file. Everything else here -- shapes,
dtypes, no-NaN -- would be caught eventually by a failing training run. Leakage
would not: it makes offline metrics *better*, so it is invisible until the model
is in production pricing real rooms with information it will never have again.

The technique used throughout is a counterfactual: build the features, mutate
only data that arrived *after* the snapshot, rebuild, and assert the features
are byte-identical. If a feature moves, it was reading the future.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import List

import numpy as np
import pandas as pd
import pytest

from database.models import Competitor, RoomType, Season
from features.feature_engineering import (
    BASE_FEATURES,
    DERIVED_FEATURES,
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    HORIZON_CHOICES,
    SEASON_FEATURES,
    TARGET_COLUMN,
    FeatureBuilder,
    FeatureConfig,
    assign_horizon,
    build_serving_row,
    describe_features,
)
from features.feature_store import (
    DERIVED_COLUMNS,
    EXOGENOUS_COLUMNS,
    FeatureStore,
    FeatureVersionMismatch,
    load_feature_list,
    save_feature_list,
    validate_feature_list,
)

STAY = date(2026, 6, 10)
HOTEL = "H001"
ROOM = RoomType.DELUXE.value
ROOM_COUNT = 40
BASE_PRICE = 6_400.0

#: Fixed horizon for the hand-built fixtures, so assertions can be exact.
FIXED_HORIZON = 7


def fixed_horizon_config(
    horizon: int = FIXED_HORIZON, *, incomplete_tail_days: int = 0
) -> FeatureConfig:
    """A config that always selects one horizon, removing sampling from tests.

    ``incomplete_tail_days=0`` because the fixtures below hand-build complete
    booking curves; the production default of 1 would discard the last night,
    which for a single-night fixture is all of them. The tail-drop behaviour has
    its own test.
    """
    return FeatureConfig(
        horizon_choices=(horizon,),
        horizon_weights=(1.0,),
        incomplete_tail_days=incomplete_tail_days,
    )


def make_bookings(
    *,
    stay: date = STAY,
    leads: List[int] = None,
    counts: List[int] = None,
    cancels: List[int] = None,
    adr: float = 6_000.0,
    hotel_id: str = HOTEL,
    room_type: str = ROOM,
) -> pd.DataFrame:
    """A booking curve: ``counts[i]`` rooms taken ``leads[i]`` days before the stay."""
    leads = [30, 21, 14, 10, 7, 5, 3, 1, 0] if leads is None else leads
    counts = [2, 3, 4, 3, 4, 5, 3, 2, 1] if counts is None else counts
    cancels = [0] * len(leads) if cancels is None else cancels

    return pd.DataFrame(
        {
            "hotel_id": hotel_id,
            "room_type": room_type,
            "booking_date": [stay - timedelta(days=lead) for lead in leads],
            "check_in_date": stay,
            "booking_count": counts,
            "cancellation_count": cancels,
            "revenue": [c * adr for c in counts],
            "adr": adr,
            "lead_time_days": leads,
        }
    )


def make_rooms(count: int = ROOM_COUNT, base_price: float = BASE_PRICE) -> pd.DataFrame:
    return pd.DataFrame(
        [{"hotel_id": HOTEL, "room_type": ROOM, "room_count": count,
          "base_price": base_price}]
    )


def make_competitor(
    *, stay: date = STAY, leads: List[int] = None, prices: List[float] = None
) -> pd.DataFrame:
    leads = [21, 7] if leads is None else leads
    prices = [6_000.0, 7_000.0] if prices is None else prices
    return pd.DataFrame(
        {
            "hotel_id": HOTEL,
            "room_type": ROOM,
            "competitor": Competitor.BOOKING.value,
            "check_in_date": stay,
            "price": prices,
            "collected_at": [
                datetime.combine(stay - timedelta(days=lead), datetime.min.time(),
                                 tzinfo=timezone.utc)
                for lead in leads
            ],
        }
    )


def make_signals(*, stay: date = STAY, search: float = 0.6) -> pd.DataFrame:
    return pd.DataFrame(
        [{"hotel_id": HOTEL, "room_type": ROOM, "stay_date": stay,
          "search_demand": search, "weather_score": 0.7, "local_event_score": 0.1}]
    )


def build(
    bookings: pd.DataFrame,
    competitor: pd.DataFrame = None,
    signals: pd.DataFrame = None,
    rooms: pd.DataFrame = None,
    config: FeatureConfig = None,
) -> pd.DataFrame:
    """Build features from hand-made frames, with a fixed horizon by default."""
    builder = FeatureBuilder(config or fixed_horizon_config())
    return builder.build(
        bookings=bookings,
        competitor_prices=competitor if competitor is not None else make_competitor(),
        signals=signals if signals is not None else make_signals(),
        rooms=rooms if rooms is not None else make_rooms(),
    )


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #


class TestFeatureContract:
    def test_columns_are_unique(self) -> None:
        assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))

    def test_target_is_not_a_feature(self) -> None:
        """The single most embarrassing bug in supervised learning."""
        assert TARGET_COLUMN not in FEATURE_COLUMNS

    def test_contract_is_the_sum_of_its_groups(self) -> None:
        assert FEATURE_COLUMNS == BASE_FEATURES + SEASON_FEATURES + DERIVED_FEATURES

    def test_every_specified_feature_is_present(self) -> None:
        """Requirement 10's list, checked literally."""
        required = {
            "occupancy_rate", "available_rooms", "total_rooms", "booking_count",
            "cancellation_count", "competitor_rate", "competitor_min_rate",
            "competitor_max_rate", "days_to_checkin", "lead_time", "search_demand",
            "historical_demand", "current_room_price", "is_weekend", "day_of_week",
            "holiday_flag", "local_event_score", "weather_score",
        }
        assert required <= set(FEATURE_COLUMNS)

    def test_season_is_one_hot_not_ordinal(self) -> None:
        """An integer season code invites a tree to split "summer > winter"."""
        assert len(SEASON_FEATURES) == len(Season)
        assert "season" not in FEATURE_COLUMNS

    def test_description_covers_every_feature(self) -> None:
        assert set(describe_features()["feature"]) == set(FEATURE_COLUMNS)


class TestHorizonAssignment:
    def test_deterministic_for_the_same_key(self) -> None:
        config = FeatureConfig()
        first = assign_horizon(HOTEL, ROOM, STAY, config)
        second = assign_horizon(HOTEL, ROOM, STAY, config)
        assert first == second

    def test_always_a_configured_horizon(self) -> None:
        config = FeatureConfig()
        for offset in range(400):
            horizon = assign_horizon(HOTEL, ROOM, STAY + timedelta(days=offset), config)
            assert horizon in HORIZON_CHOICES

    def test_distribution_follows_the_weights(self) -> None:
        """Concentrated where pricing decisions actually get made."""
        config = FeatureConfig()
        horizons = [
            assign_horizon(HOTEL, ROOM, STAY + timedelta(days=n), config)
            for n in range(2_000)
        ]
        short = sum(1 for h in horizons if h <= 14) / len(horizons)
        assert 0.55 < short < 0.85

    def test_different_keys_get_different_horizons(self) -> None:
        config = FeatureConfig()
        assert len({
            assign_horizon(HOTEL, ROOM, STAY + timedelta(days=n), config)
            for n in range(50)
        }) > 3


# --------------------------------------------------------------------------- #
# Booking-curve arithmetic
# --------------------------------------------------------------------------- #


class TestBookingCurve:
    def test_occupancy_counts_only_bookings_made_before_the_snapshot(self) -> None:
        """Leads 30, 21, 14, 10 and 7 are on the books at a 7-day horizon."""
        frame = build(make_bookings())
        row = frame.iloc[0]
        assert row["occupancy_rate"] == pytest.approx((2 + 3 + 4 + 3 + 4) / ROOM_COUNT)
        assert row["available_rooms"] == ROOM_COUNT - 16

    def test_target_uses_the_final_net_total(self) -> None:
        frame = build(make_bookings())
        assert frame.iloc[0][TARGET_COLUMN] == pytest.approx(27 / ROOM_COUNT)

    def test_cancellations_are_excluded_from_occupancy(self) -> None:
        """The schema records which booking a cancellation came from, not when
        it happened -- netting it off at snapshot time would use the future."""
        gross = build(make_bookings())
        netted = build(make_bookings(cancels=[1, 1, 1, 1, 1, 0, 0, 0, 0]))
        assert netted.iloc[0]["occupancy_rate"] == gross.iloc[0]["occupancy_rate"]
        assert netted.iloc[0][TARGET_COLUMN] < gross.iloc[0][TARGET_COLUMN]

    def test_pickup_is_the_last_seven_days_of_bookings(self) -> None:
        """Leads 14 and 10 fall in the window [7, 14); leads 21 and 30 do not."""
        frame = build(make_bookings())
        assert frame.iloc[0]["booking_count"] == pytest.approx(4 + 3)

    def test_pickup_velocity_is_rooms_per_day(self) -> None:
        frame = build(make_bookings())
        assert frame.iloc[0]["pickup_velocity"] == pytest.approx(7 / 7)

    def test_empty_book_falls_back_to_base_price_and_horizon(self) -> None:
        """Nothing on the books yet: no achieved rate exists to average."""
        frame = build(make_bookings(leads=[3, 1], counts=[5, 5]))
        row = frame.iloc[0]
        assert row["occupancy_rate"] == 0.0
        assert row["current_room_price"] == pytest.approx(BASE_PRICE)
        assert row["lead_time"] == pytest.approx(FIXED_HORIZON)

    def test_occupancy_is_capped_at_full(self) -> None:
        """Overbooking is real, but a hotel cannot be 130% occupied on the books."""
        frame = build(make_bookings(leads=[30], counts=[ROOM_COUNT + 20]))
        assert frame.iloc[0]["occupancy_rate"] == 1.0
        assert frame.iloc[0]["available_rooms"] == 0.0

    def test_overbooking_survives_in_the_target(self) -> None:
        """'We sold 110% of the rooms' is information, not an error to clip away."""
        frame = build(make_bookings(leads=[30], counts=[int(ROOM_COUNT * 1.1)]))
        assert frame.iloc[0][TARGET_COLUMN] > 1.0

    def test_mean_lead_time_is_booking_weighted(self) -> None:
        frame = build(make_bookings(leads=[30, 10], counts=[1, 3]))
        # (30*1 + 10*3) / 4
        assert frame.iloc[0]["lead_time"] == pytest.approx(15.0)

    def test_occupancy_grows_as_check_in_approaches(self) -> None:
        """The booking curve, asserted as a shape rather than a single number.

        Regression test for a real defect. ``merge_asof`` returns a fresh
        ``RangeIndex``, so restoring row order with ``sort_index()`` silently
        re-sorted by *position* and handed every row some other row's
        on-the-books total. The column still looked plausible in aggregate --
        right range, right mean -- but occupancy came out flat at ~0.48 across
        every horizon, and the trained model ranked occupancy near-worthless.
        """
        stays = [STAY - timedelta(days=n) for n in range(30)]
        bookings = pd.concat([make_bookings(stay=s) for s in stays], ignore_index=True)
        signals = pd.concat([make_signals(stay=s) for s in stays], ignore_index=True)
        competitor = pd.concat([make_competitor(stay=s) for s in stays], ignore_index=True)

        # Default curve: 2 rooms at 30 days out, 9 by 14 days, 16 by 7, 27 total.
        expected = {0: 27, 7: 16, 14: 9, 30: 2}

        for horizon, rooms in expected.items():
            frame = build(bookings, competitor, signals,
                          config=fixed_horizon_config(horizon))
            assert frame["occupancy_rate"].mean() == pytest.approx(
                rooms / ROOM_COUNT
            ), f"horizon {horizon}"

    def test_each_row_gets_its_own_booking_curve(self) -> None:
        """Row-level alignment under *varying* horizons.

        The companion test above uses one fixed horizon, which a stable sort
        leaves in place -- so it cannot see a misordered join. This one uses the
        real sampled-horizon config, which genuinely reorders rows inside
        ``merge_asof``, and checks each row against its own curve computed
        independently. Every stay date is given a different booking volume so a
        permutation of the right answers is not a passing result.
        """
        stays = [STAY - timedelta(days=n) for n in range(1, 40)]
        volumes = {s: 2 + (index * 3) % 34 for index, s in enumerate(stays)}

        bookings = pd.concat(
            [
                make_bookings(stay=s, leads=[45], counts=[volumes[s]])
                for s in stays
            ],
            ignore_index=True,
        )
        signals = pd.concat([make_signals(stay=s) for s in stays], ignore_index=True)
        competitor = pd.concat([make_competitor(stay=s) for s in stays], ignore_index=True)

        # The real config: horizons vary per row, so the internal sort reorders.
        frame = build(bookings, competitor, signals, config=FeatureConfig(
            incomplete_tail_days=0
        ))
        assert frame["days_to_checkin"].nunique() > 3, "horizons did not vary"

        for row in frame.itertuples():
            stay = row.stay_date.date()
            # Every booking sits at lead 45, so it is on the books at any
            # horizon at or below 45 and invisible beyond it.
            on_books = volumes[stay] if row.days_to_checkin <= 45 else 0
            assert row.occupancy_rate == pytest.approx(on_books / ROOM_COUNT), (
                f"{stay} at horizon {row.days_to_checkin}"
            )


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #


class TestNoLeakage:
    """Counterfactual tests: change only the future, assert the features do not."""

    def test_bookings_after_the_snapshot_do_not_move_the_features(self) -> None:
        baseline = build(make_bookings())
        # Same curve, but far more rooms sold in the final week.
        surge = build(make_bookings(counts=[2, 3, 4, 3, 4, 40, 40, 40, 40]))

        for column in FEATURE_COLUMNS:
            assert baseline.iloc[0][column] == pytest.approx(surge.iloc[0][column]), (
                f"{column} changed when only post-snapshot bookings changed"
            )
        # The target must move, or the counterfactual proved nothing.
        assert surge.iloc[0][TARGET_COLUMN] > baseline.iloc[0][TARGET_COLUMN]

    def test_competitor_rates_collected_after_the_snapshot_are_invisible(self) -> None:
        baseline = build(make_bookings(), competitor=make_competitor())
        # A wild rate published two days before check-in, after our snapshot.
        later = build(
            make_bookings(),
            competitor=make_competitor(leads=[21, 7, 2], prices=[6_000, 7_000, 99_000]),
        )
        assert baseline.iloc[0]["competitor_rate"] == later.iloc[0]["competitor_rate"]
        assert baseline.iloc[0]["competitor_max_rate"] == later.iloc[0]["competitor_max_rate"]

    def test_the_freshest_visible_rate_wins(self) -> None:
        """Observations at 21 and 7 days out; at a 7-day horizon both are visible
        and the 7-day one is the current view of the market."""
        frame = build(make_bookings(), competitor=make_competitor())
        assert frame.iloc[0]["competitor_rate"] == pytest.approx(7_000.0)

    def test_history_window_ends_at_the_snapshot_not_the_stay(self) -> None:
        """A row priced 30 days out must see a month less history than a
        same-day row, exactly as it would in production."""
        stays = [STAY - timedelta(days=n) for n in range(40, -1, -1)]
        bookings = pd.concat(
            [make_bookings(stay=s, leads=[40], counts=[10 + i])
             for i, s in enumerate(stays)],
            ignore_index=True,
        )
        signals = pd.concat([make_signals(stay=s) for s in stays], ignore_index=True)
        competitor = pd.concat([make_competitor(stay=s) for s in stays], ignore_index=True)

        near = build(bookings, competitor, signals, config=fixed_horizon_config(0))
        far = build(bookings, competitor, signals, config=fixed_horizon_config(30))

        target_stay = pd.Timestamp(STAY - timedelta(days=1))
        near_row = near[near["stay_date"] == target_stay].iloc[0]
        far_row = far[far["stay_date"] == target_stay].iloc[0]

        # Demand is rising over the window, so a window ending 30 days earlier
        # must see a lower average.
        assert far_row["historical_demand"] < near_row["historical_demand"]

    def test_target_is_absent_from_the_feature_block(self) -> None:
        frame = build(make_bookings())
        assert TARGET_COLUMN not in frame[list(FEATURE_COLUMNS)].columns

    def test_incomplete_tail_rows_are_dropped(self) -> None:
        """The last night in the data is still taking bookings; its 'final'
        total is not final."""
        stays = [STAY - timedelta(days=n) for n in range(4)]
        bookings = pd.concat(
            [make_bookings(stay=s) for s in stays], ignore_index=True
        )
        signals = pd.concat([make_signals(stay=s) for s in stays], ignore_index=True)
        competitor = pd.concat([make_competitor(stay=s) for s in stays], ignore_index=True)

        frame = build(
            bookings, competitor, signals,
            config=fixed_horizon_config(incomplete_tail_days=1),
        )
        assert pd.Timestamp(STAY) not in set(frame["stay_date"])
        assert len(frame) == 3

    def test_all_nights_incomplete_is_an_actionable_error(self) -> None:
        """A single night with the tail filter on leaves nothing to learn from,
        and should say so rather than fail deep inside a merge."""
        with pytest.raises(ValueError, match="incomplete-tail filter"):
            build(
                make_bookings(),
                config=fixed_horizon_config(incomplete_tail_days=1),
            )


# --------------------------------------------------------------------------- #
# Missing data and edge cases
# --------------------------------------------------------------------------- #


class TestMissingData:
    def test_absent_competitor_data_falls_back_to_base_price(self) -> None:
        """A long horizon can legitimately see no market at all."""
        frame = build(make_bookings(), competitor=make_competitor(leads=[3], prices=[9_000]))
        row = frame.iloc[0]
        assert row["competitor_missing"] == 1.0
        assert row["competitor_rate"] == pytest.approx(BASE_PRICE)
        assert row["competitor_count"] == 0.0

    def test_present_competitor_data_clears_the_flag(self) -> None:
        assert build(make_bookings()).iloc[0]["competitor_missing"] == 0.0

    def test_completely_empty_competitor_frame_is_survivable(self) -> None:
        frame = build(make_bookings(), competitor=pd.DataFrame())
        assert frame.iloc[0]["competitor_missing"] == 1.0
        assert frame.iloc[0]["competitor_rate"] == pytest.approx(BASE_PRICE)

    def test_missing_signals_use_neutral_defaults(self) -> None:
        frame = build(make_bookings(), signals=pd.DataFrame())
        row = frame.iloc[0]
        assert row["search_demand"] == 0.0
        assert row["weather_score"] == 0.5

    def test_zero_row_bookings_are_ignored(self) -> None:
        """A pickup row with no rooms would only add zero weight to every
        weighted average downstream."""
        with_empties = make_bookings(
            leads=[30, 25, 14, 7], counts=[5, 0, 5, 5]
        )
        frame = build(with_empties)
        assert frame.iloc[0]["occupancy_rate"] == pytest.approx(15 / ROOM_COUNT)

    def test_no_nulls_or_infinities_reach_the_model(self) -> None:
        frame = build(make_bookings())
        block = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        assert np.isfinite(block).all()

    def test_pipeline_refuses_to_emit_a_null(self, monkeypatch) -> None:
        """A NaN reaching a model is a silent failure; make it loud."""
        original = FeatureBuilder._add_derived_features

        def _sabotage(frame: pd.DataFrame) -> pd.DataFrame:
            frame = original(frame)
            frame.loc[frame.index[0], "demand_pressure"] = np.nan
            return frame

        monkeypatch.setattr(FeatureBuilder, "_add_derived_features",
                            staticmethod(_sabotage))
        with pytest.raises(ValueError, match="null values"):
            build(make_bookings())

    def test_pipeline_refuses_to_emit_an_infinity(self, monkeypatch) -> None:
        original = FeatureBuilder._add_derived_features

        def _sabotage(frame: pd.DataFrame) -> pd.DataFrame:
            frame = original(frame)
            frame.loc[frame.index[0], "demand_pressure"] = np.inf
            return frame

        monkeypatch.setattr(FeatureBuilder, "_add_derived_features",
                            staticmethod(_sabotage))
        with pytest.raises(ValueError, match="non-finite"):
            build(make_bookings())


class TestCalendarFeatures:
    def test_weekend_and_day_of_week_agree(self) -> None:
        saturday = date(2026, 6, 13)
        stays = [saturday, saturday - timedelta(days=1)]
        bookings = pd.concat([make_bookings(stay=s) for s in stays], ignore_index=True)
        signals = pd.concat([make_signals(stay=s) for s in stays], ignore_index=True)
        competitor = pd.concat([make_competitor(stay=s) for s in stays], ignore_index=True)

        frame = build(bookings, competitor, signals).set_index("stay_date")
        friday = frame.loc[pd.Timestamp(saturday - timedelta(days=1))]
        assert friday["day_of_week"] == 4
        assert friday["is_weekend"] == 0.0

    def test_season_one_hot_sums_to_one(self) -> None:
        frame = build(make_bookings())
        assert frame[list(SEASON_FEATURES)].sum(axis=1).eq(1.0).all()

    def test_monsoon_stay_is_flagged_monsoon(self) -> None:
        frame = build(make_bookings())  # 10 June
        assert frame.iloc[0]["season_monsoon"] == 1.0
        assert frame.iloc[0]["season_winter"] == 0.0


# --------------------------------------------------------------------------- #
# Train/serve parity
# --------------------------------------------------------------------------- #


class TestServingParity:
    def _serving_row(self, **overrides) -> pd.DataFrame:
        kwargs = dict(
            hotel_id=HOTEL,
            room_type=RoomType.DELUXE,
            check_in_date=STAY,
            base_price=BASE_PRICE,
            total_rooms=ROOM_COUNT,
            as_of=STAY - timedelta(days=FIXED_HORIZON),
        )
        kwargs.update(overrides)
        return build_serving_row(**kwargs)

    def test_produces_exactly_the_training_columns_in_order(self) -> None:
        """The seam where train/serve skew would appear."""
        assert list(self._serving_row().columns) == list(FEATURE_COLUMNS)

    def test_single_finite_row(self) -> None:
        row = self._serving_row()
        assert len(row) == 1
        assert np.isfinite(row.to_numpy(dtype=float)).all()

    def test_horizon_comes_from_the_pricing_date(self) -> None:
        assert self._serving_row().iloc[0]["days_to_checkin"] == FIXED_HORIZON

    def test_past_check_in_clamps_the_horizon_to_zero(self) -> None:
        row = self._serving_row(as_of=STAY + timedelta(days=5))
        assert row.iloc[0]["days_to_checkin"] == 0.0

    def test_occupancy_derived_from_available_rooms(self) -> None:
        row = self._serving_row(available_rooms=10, occupancy_rate=None)
        assert row.iloc[0]["occupancy_rate"] == pytest.approx(0.75)

    def test_available_rooms_derived_from_occupancy(self) -> None:
        row = self._serving_row(occupancy_rate=0.75)
        assert row.iloc[0]["available_rooms"] == pytest.approx(10.0)

    def test_absent_competitor_rate_is_flagged_and_imputed(self) -> None:
        row = self._serving_row()
        assert row.iloc[0]["competitor_missing"] == 1.0
        assert row.iloc[0]["competitor_rate"] == pytest.approx(BASE_PRICE)

    def test_supplied_competitor_rate_clears_the_flag(self) -> None:
        row = self._serving_row(competitor_rate=7_000.0)
        assert row.iloc[0]["competitor_missing"] == 0.0
        assert row.iloc[0]["competitor_rate"] == pytest.approx(7_000.0)

    def test_derived_features_match_the_training_arithmetic(self) -> None:
        """The same inputs must give the same derived values on both paths."""
        training = build(make_bookings()).iloc[0]
        serving = self._serving_row(
            occupancy_rate=float(training["occupancy_rate"]),
            available_rooms=int(training["available_rooms"]),
            current_price=float(training["current_room_price"]),
            competitor_rate=float(training["competitor_rate"]),
            competitor_min_rate=float(training["competitor_min_rate"]),
            competitor_max_rate=float(training["competitor_max_rate"]),
            booking_count=float(training["booking_count"]),
            search_demand=float(training["search_demand"]),
            local_event_score=float(training["local_event_score"]),
        ).iloc[0]

        for column in DERIVED_FEATURES:
            if column == "competitor_missing":
                continue
            assert serving[column] == pytest.approx(training[column]), column

    def test_calendar_features_match_the_training_path(self) -> None:
        training = build(make_bookings()).iloc[0]
        serving = self._serving_row().iloc[0]
        for column in ("is_weekend", "day_of_week", "holiday_flag",
                       "holiday_proximity", *SEASON_FEATURES):
            assert serving[column] == pytest.approx(training[column]), column


# --------------------------------------------------------------------------- #
# The feature list artifact
# --------------------------------------------------------------------------- #


class TestFeatureListArtifact:
    def test_round_trip(self, tmp_path) -> None:
        save_feature_list(tmp_path)
        saved = load_feature_list(tmp_path)
        assert saved["features"] == list(FEATURE_COLUMNS)
        assert saved["feature_version"] == FEATURE_VERSION
        validate_feature_list(saved)

    def test_missing_file_gives_an_actionable_error(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="train_models"):
            load_feature_list(tmp_path / "absent")

    def test_added_feature_is_rejected(self, tmp_path) -> None:
        save_feature_list(tmp_path)
        saved = load_feature_list(tmp_path)
        with pytest.raises(FeatureVersionMismatch, match="New:"):
            validate_feature_list(saved, columns=list(FEATURE_COLUMNS) + ["new_thing"])

    def test_removed_feature_is_rejected(self, tmp_path) -> None:
        save_feature_list(tmp_path)
        saved = load_feature_list(tmp_path)
        with pytest.raises(FeatureVersionMismatch, match="Missing now:"):
            validate_feature_list(saved, columns=list(FEATURE_COLUMNS)[:-1])

    def test_reordered_features_are_rejected(self, tmp_path) -> None:
        """A positionally-indexed model would read every column shifted by one
        and produce plausible, wrong numbers with no error anywhere."""
        save_feature_list(tmp_path)
        saved = load_feature_list(tmp_path)
        shuffled = [FEATURE_COLUMNS[1], FEATURE_COLUMNS[0], *FEATURE_COLUMNS[2:]]
        with pytest.raises(FeatureVersionMismatch, match="order differs"):
            validate_feature_list(saved, columns=shuffled)


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #


class TestFeatureStore:
    def _seed_raw(self, session) -> None:
        """Insert a small but complete raw dataset for one hotel."""
        from database.models import Booking, CompetitorPrice, DemandFeature

        stays = [STAY - timedelta(days=n) for n in range(10, 0, -1)]
        for stay in stays:
            for lead, count in ((21, 4), (14, 5), (7, 6), (2, 5)):
                session.add(
                    Booking(
                        hotel_id=HOTEL,
                        room_type=RoomType.DELUXE,
                        booking_date=stay - timedelta(days=lead),
                        check_in_date=stay,
                        check_out_date=stay + timedelta(days=1),
                        booking_count=count,
                        cancellation_count=0,
                        revenue=count * 6_000.0,
                        adr=6_000.0,
                        lead_time_days=lead,
                    )
                )
            session.add(
                CompetitorPrice(
                    hotel_id=HOTEL,
                    room_type=RoomType.DELUXE,
                    competitor=Competitor.BOOKING,
                    check_in_date=stay,
                    price=6_800.0,
                    collected_at=datetime.combine(
                        stay - timedelta(days=10), datetime.min.time(),
                        tzinfo=timezone.utc
                    ),
                )
            )
            session.add(
                DemandFeature(
                    hotel_id=HOTEL,
                    room_type=RoomType.DELUXE,
                    stay_date=stay,
                    day_of_week=stay.weekday(),
                    is_weekend=stay.weekday() >= 5,
                    season=Season.MONSOON,
                    holiday_flag=False,
                    local_event_score=0.2,
                    weather_score=0.6,
                    search_demand=0.55,
                )
            )
        session.commit()

    def test_build_and_store_round_trip(self, seeded_session) -> None:
        self._seed_raw(seeded_session)
        store = FeatureStore(fixed_horizon_config())

        frame = store.build_and_store(seeded_session)
        assert not frame.empty

        loaded = store.load_training_frame(seeded_session)
        assert len(loaded) == len(frame)
        assert loaded["feature_version"].eq(FEATURE_VERSION).all()
        assert loaded["target_demand"].notna().all()

    def test_writing_features_does_not_clobber_exogenous_signals(
        self, seeded_session
    ) -> None:
        """The streaming consumer owns those columns; a rebuild must not touch
        them."""
        from database.models import DemandFeature

        self._seed_raw(seeded_session)
        store = FeatureStore(fixed_horizon_config())
        store.build_and_store(seeded_session)

        rows = seeded_session.query(DemandFeature).all()
        assert all(row.search_demand == pytest.approx(0.55) for row in rows)
        assert all(row.weather_score == pytest.approx(0.6) for row in rows)

    def test_derived_columns_are_populated(self, seeded_session) -> None:
        from database.models import DemandFeature

        self._seed_raw(seeded_session)
        FeatureStore(fixed_horizon_config()).build_and_store(seeded_session)

        row = seeded_session.query(DemandFeature).filter(
            DemandFeature.target_demand.is_not(None)
        ).first()
        assert row.occupancy_rate is not None
        assert row.competitor_rate is not None
        assert row.days_to_checkin == FIXED_HORIZON
        assert row.feature_version == FEATURE_VERSION

    def test_rebuild_is_idempotent(self, seeded_session) -> None:
        self._seed_raw(seeded_session)
        store = FeatureStore(fixed_horizon_config())

        first = store.build_and_store(seeded_session)
        second = store.build_and_store(seeded_session)

        assert len(store.load_training_frame(seeded_session)) == len(first)
        pd.testing.assert_series_equal(
            first[TARGET_COLUMN], second[TARGET_COLUMN], check_names=False
        )

    def test_empty_database_gives_an_actionable_error(self, db_session) -> None:
        with pytest.raises(ValueError, match="seed_database"):
            FeatureStore().build(db_session)

    def test_coverage_reports_progress(self, seeded_session) -> None:
        self._seed_raw(seeded_session)
        store = FeatureStore(fixed_horizon_config())

        before = store.coverage(seeded_session)
        assert before["computed"] == 0

        store.build_and_store(seeded_session)
        after = store.coverage(seeded_session)
        assert after["computed"] > 0
        assert after["labelled"] == after["computed"]

    def test_latest_for_returns_one_night(self, seeded_session) -> None:
        self._seed_raw(seeded_session)
        FeatureStore(fixed_horizon_config()).build_and_store(seeded_session)

        row = FeatureStore.latest_for(
            seeded_session, HOTEL, RoomType.DELUXE, STAY - timedelta(days=5)
        )
        assert row is not None
        assert row["occupancy_rate"] is not None

    def test_latest_for_returns_none_when_absent(self, seeded_session) -> None:
        assert FeatureStore.latest_for(
            seeded_session, HOTEL, RoomType.DELUXE, date(2030, 1, 1)
        ) is None

    def test_owned_columns_do_not_overlap(self) -> None:
        """Two writers, two disjoint column sets. That is the whole design."""
        assert not set(DERIVED_COLUMNS) & set(EXOGENOUS_COLUMNS)

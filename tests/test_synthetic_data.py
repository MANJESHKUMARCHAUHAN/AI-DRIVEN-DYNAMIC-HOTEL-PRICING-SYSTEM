"""Tests for the calendar and the synthetic data generator.

Two categories, and the second is the interesting one:

**Validity** -- nothing negative, nothing impossible, referential integrity
holds, the same seed reproduces the same bytes.

**Realism** -- the relationships requirement 7 demands are actually present in
the numbers. These are statistical assertions on generated data, which sounds
fragile but is not: the generator is seeded, so the values are fixed, and the
thresholds are set well inside the observed margins. If someone later "tidies
up" the demand model and flattens the weekend effect, these tests fail -- which
is exactly what should happen, because every downstream model would silently get
worse instead.

The realism suite deliberately correlates *within* ``(hotel, room_type)``. Pooled
across room types the occupancy/price relationship inverts, because suites are
the least occupied and the most expensive category -- a textbook Simpson's
paradox, and a trap worth documenting rather than accidentally asserting.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from database.models import Competitor, MarketSegment, RoomType, Season
from features.calendars import (
    event_score,
    events_on,
    holiday_name,
    holiday_on,
    holiday_proximity,
    is_holiday,
    is_weekend,
    known_event_cities,
    season_of,
)
from ingestion.synthetic_dataset import (
    CITY_PROFILES,
    COMPETITOR_OBSERVATION_LEADS,
    HOTEL_CATALOG,
    MAX_LEAD_DAYS,
    ROOM_MIX,
    SyntheticDataset,
    SyntheticDatasetGenerator,
    generate_dataset,
    summarise,
)

#: Fixed end date so the generated window -- and therefore every statistic
#: asserted below -- does not drift with the wall clock.
END_DATE = date(2026, 6, 30)

#: Four hotels is the smallest slice of the catalogue that still covers all
#: three market segments and includes Goa, whose seasonality is the strongest
#: signal in the dataset.
N_HOTELS = 4


@pytest.fixture(scope="module")
def dataset() -> SyntheticDataset:
    """One year of data for four hotels. Generated once for the whole module."""
    return generate_dataset(
        seed=42, n_hotels=N_HOTELS, history_days=365, end_date=END_DATE
    )


@pytest.fixture(scope="module")
def nights(dataset: SyntheticDataset) -> pd.DataFrame:
    """One row per (hotel, room type, night) with occupancy, ADR and rates.

    This is the shape the realism assertions need: the generator emits pickup
    rows keyed on *booking* date, and occupancy is a property of the *stay*.
    """
    bookings = dataset.bookings.assign(
        sold=lambda d: d.booking_count - d.cancellation_count
    )
    frame = (
        bookings.groupby(["hotel_id", "room_type", "check_in_date"])
        .agg(sold=("sold", "sum"), revenue=("revenue", "sum"),
             gross=("booking_count", "sum"), cancels=("cancellation_count", "sum"))
        .reset_index()
        .merge(dataset.rooms[["hotel_id", "room_type", "room_count"]],
               on=["hotel_id", "room_type"])
        .merge(dataset.hotels[["hotel_id", "city", "segment"]], on="hotel_id")
    )
    frame["occupancy"] = frame["sold"] / frame["room_count"]
    frame["adr"] = frame["revenue"] / frame["sold"].replace(0, np.nan)
    frame["dow"] = pd.to_datetime(frame["check_in_date"]).dt.dayofweek
    frame["month"] = pd.to_datetime(frame["check_in_date"]).dt.month
    frame["is_weekend"] = frame["dow"] >= 5

    competitor = (
        dataset.competitor_prices.groupby(["hotel_id", "room_type", "check_in_date"])
        .agg(competitor_rate=("price", "mean"),
             competitor_min=("price", "min"),
             competitor_max=("price", "max"),
             competitor_count=("price", "size"))
        .reset_index()
    )
    frame = frame.merge(
        competitor, on=["hotel_id", "room_type", "check_in_date"], how="left"
    )
    return frame.merge(
        dataset.demand_signals,
        left_on=["hotel_id", "room_type", "check_in_date"],
        right_on=["hotel_id", "room_type", "stay_date"],
        how="left",
        suffixes=("", "_signal"),
    )


def _within_group_corr(frame: pd.DataFrame, left: str, right: str) -> pd.Series:
    """Correlation of two columns computed inside each (hotel, room type)."""
    return frame.groupby(["hotel_id", "room_type"]).apply(
        lambda d: d[left].corr(d[right])
    )


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #


class TestCalendar:
    @pytest.mark.parametrize(
        ("day", "expected"),
        [
            (date(2026, 1, 15), Season.WINTER),
            (date(2026, 4, 10), Season.SUMMER),
            (date(2026, 7, 20), Season.MONSOON),
            (date(2026, 11, 3), Season.AUTUMN),
            (date(2026, 12, 31), Season.WINTER),
        ],
    )
    def test_season_mapping(self, day: date, expected: Season) -> None:
        assert season_of(day) is expected

    def test_every_month_maps_to_a_season(self) -> None:
        assert {season_of(date(2026, m, 1)) for m in range(1, 13)} == set(Season)

    def test_weekend_is_saturday_and_sunday(self) -> None:
        monday = date(2026, 6, 1)
        flags = [is_weekend(monday + timedelta(days=n)) for n in range(7)]
        assert flags == [False, False, False, False, False, True, True]

    def test_known_holidays_are_recognised(self) -> None:
        assert holiday_name(date(2026, 11, 8)) == "Diwali"
        assert holiday_name(date(2026, 8, 15)) == "Independence Day"
        assert holiday_name(date(2026, 1, 26)) == "Republic Day"

    def test_ordinary_day_is_not_a_holiday(self) -> None:
        assert holiday_on(date(2026, 6, 17)) is None
        assert is_holiday(date(2026, 6, 17)) is False

    def test_fixed_holidays_recur_every_year(self) -> None:
        for year in (2024, 2025, 2026, 2027):
            assert holiday_name(date(year, 12, 25)) == "Christmas"

    def test_holiday_pressure_decays_with_distance(self) -> None:
        diwali = date(2026, 11, 8)
        on_the_day = holiday_proximity(diwali)
        one_day_out = holiday_proximity(diwali + timedelta(days=1))
        far_away = holiday_proximity(diwali + timedelta(days=10))
        assert on_the_day > one_day_out > far_away
        assert far_away == 0.0

    def test_holiday_pressure_is_bounded(self) -> None:
        day = date(2026, 1, 1)
        scores = [holiday_proximity(day + timedelta(days=n)) for n in range(365)]
        assert min(scores) >= 0.0
        assert max(scores) <= 1.0

    def test_event_windows_resolve(self) -> None:
        assert "Jaipur Literature Festival" in [
            e.name for e in events_on("Jaipur", date(2026, 1, 24))
        ]
        assert events_on("Jaipur", date(2026, 6, 15)) == []

    def test_event_window_wrapping_the_year_end(self) -> None:
        """Goa's New Year week runs 26 December to 2 January."""
        assert event_score("Goa", date(2026, 12, 28)) > 0.9
        assert event_score("Goa", date(2026, 1, 1)) > 0.9
        assert event_score("Goa", date(2026, 1, 10)) == 0.0

    def test_overlapping_events_combine_without_exceeding_one(self) -> None:
        """Sunburn during New Year week: busier than either, never over 100%."""
        combined = event_score("Goa", date(2026, 12, 28))
        assert 0.9 < combined <= 1.0

    def test_event_score_is_zero_for_unknown_cities(self) -> None:
        assert event_score("Atlantis", date(2026, 1, 1)) == 0.0

    def test_generated_hotels_live_in_cities_the_calendar_knows(self) -> None:
        """A hotel in a city with no event calendar would train on a constant."""
        assert {h.city for h in HOTEL_CATALOG} <= known_event_cities()

    def test_biennial_events_skip_odd_years(self) -> None:
        assert event_score("New Delhi", date(2026, 1, 15)) > 0.0
        assert event_score("New Delhi", date(2027, 1, 15)) == 0.0


# --------------------------------------------------------------------------- #
# Structure and validity
# --------------------------------------------------------------------------- #


class TestStructure:
    def test_all_five_frames_are_populated(self, dataset: SyntheticDataset) -> None:
        counts = dataset.row_counts()
        assert counts["hotels"] == N_HOTELS
        assert counts["rooms"] == N_HOTELS * len(ROOM_MIX)
        assert counts["bookings"] > 50_000
        assert counts["competitor_prices"] > 10_000
        assert counts["demand_signals"] == N_HOTELS * len(ROOM_MIX) * 365

    def test_room_counts_sum_to_hotel_inventory(self, dataset: SyntheticDataset) -> None:
        totals = dataset.rooms.groupby("hotel_id").room_count.sum()
        declared = dataset.hotels.set_index("hotel_id").total_rooms
        assert (totals == declared).all()

    def test_every_hotel_sells_every_room_type(self, dataset: SyntheticDataset) -> None:
        per_hotel = dataset.rooms.groupby("hotel_id").room_type.nunique()
        assert (per_hotel == len(RoomType)).all()

    def test_room_mix_shares_sum_to_one(self) -> None:
        assert sum(spec.inventory_share for spec in ROOM_MIX) == pytest.approx(1.0)

    def test_base_price_rises_with_room_class(self, dataset: SyntheticDataset) -> None:
        order = [rt.value for rt in RoomType]
        prices = (
            dataset.rooms[dataset.rooms.hotel_id == "H001"]
            .set_index("room_type")
            .base_price.reindex(order)
        )
        assert prices.is_monotonic_increasing

    def test_every_generated_city_has_a_profile(self) -> None:
        assert {h.city for h in HOTEL_CATALOG} <= set(CITY_PROFILES)

    def test_referential_integrity_of_fact_frames(self, dataset: SyntheticDataset) -> None:
        rooms = set(zip(dataset.rooms.hotel_id, dataset.rooms.room_type))
        for name in ("bookings", "competitor_prices", "demand_signals"):
            frame = dataset.frames()[name]
            keys = set(zip(frame.hotel_id, frame.room_type))
            assert keys <= rooms, f"{name} references a room that does not exist"


class TestValidity:
    def test_no_negative_quantities(self, dataset: SyntheticDataset) -> None:
        bookings = dataset.bookings
        assert (bookings.booking_count >= 0).all()
        assert (bookings.cancellation_count >= 0).all()
        assert (bookings.revenue >= 0).all()
        assert (bookings.adr > 0).all()
        assert (bookings.lead_time_days >= 0).all()

    def test_cancellations_never_exceed_bookings(self, dataset: SyntheticDataset) -> None:
        assert (dataset.bookings.cancellation_count <= dataset.bookings.booking_count).all()

    def test_bookings_precede_the_stay(self, dataset: SyntheticDataset) -> None:
        assert (dataset.bookings.booking_date <= dataset.bookings.check_in_date).all()

    def test_lead_time_matches_the_dates(self, dataset: SyntheticDataset) -> None:
        gap = (
            pd.to_datetime(dataset.bookings.check_in_date)
            - pd.to_datetime(dataset.bookings.booking_date)
        ).dt.days
        assert (gap == dataset.bookings.lead_time_days).all()
        assert dataset.bookings.lead_time_days.max() <= MAX_LEAD_DAYS

    def test_grain_is_one_room_night(self, dataset: SyntheticDataset) -> None:
        nights_stayed = (
            pd.to_datetime(dataset.bookings.check_out_date)
            - pd.to_datetime(dataset.bookings.check_in_date)
        ).dt.days
        assert (nights_stayed == 1).all()

    def test_competitor_prices_are_positive_and_in_configured_currency(
        self, dataset: SyntheticDataset
    ) -> None:
        assert (dataset.competitor_prices.price > 0).all()
        assert dataset.competitor_prices.currency.nunique() == 1

    def test_competitor_observations_land_on_the_configured_leads(
        self, dataset: SyntheticDataset
    ) -> None:
        leads = (
            pd.to_datetime(dataset.competitor_prices.check_in_date)
            - dataset.competitor_prices.collected_at.dt.tz_localize(None).dt.normalize()
        ).dt.days
        assert set(leads.unique()) <= set(COMPETITOR_OBSERVATION_LEADS)

    def test_signal_scores_are_bounded(self, dataset: SyntheticDataset) -> None:
        signals = dataset.demand_signals
        for column in ("local_event_score", "weather_score", "search_demand"):
            assert signals[column].between(0.0, 1.0).all(), column

    def test_calendar_columns_agree_with_the_calendar_module(
        self, dataset: SyntheticDataset
    ) -> None:
        """Train/serve parity starts here: one definition of ``season``."""
        sample = dataset.demand_signals.sample(200, random_state=0)
        for row in sample.itertuples():
            assert row.season == season_of(row.stay_date).value
            assert row.is_weekend == is_weekend(row.stay_date)
            assert row.holiday_flag == is_holiday(row.stay_date)
            assert row.day_of_week == row.stay_date.weekday()

    def test_occupancy_stays_close_to_capacity(self, nights: pd.DataFrame) -> None:
        """Overbooking is real, so a few nights exceed 100% -- but only a few."""
        assert nights.occupancy.min() >= 0.0
        assert nights.occupancy.max() < 1.35
        assert (nights.occupancy > 1.0).mean() < 0.06


class TestDeterminism:
    def test_same_seed_reproduces_the_same_data(self) -> None:
        kwargs = dict(seed=7, n_hotels=2, history_days=60, end_date=END_DATE)
        first = generate_dataset(**kwargs)
        second = generate_dataset(**kwargs)
        for name, frame in first.frames().items():
            pd.testing.assert_frame_equal(frame, second.frames()[name])

    def test_different_seed_produces_different_data(self) -> None:
        a = generate_dataset(seed=1, n_hotels=2, history_days=60, end_date=END_DATE)
        b = generate_dataset(seed=2, n_hotels=2, history_days=60, end_date=END_DATE)
        assert not a.bookings.equals(b.bookings)

    def test_adding_a_hotel_does_not_perturb_the_others(self) -> None:
        """Per-hotel child seeds. Otherwise every dataset regenerates on a
        catalogue change and no experiment is comparable to the last one."""
        two = generate_dataset(seed=42, n_hotels=2, history_days=60, end_date=END_DATE)
        three = generate_dataset(seed=42, n_hotels=3, history_days=60, end_date=END_DATE)
        shared = three.bookings[three.bookings.hotel_id.isin(["H001", "H002"])]
        pd.testing.assert_frame_equal(
            two.bookings.reset_index(drop=True), shared.reset_index(drop=True)
        )

    def test_csv_round_trip_preserves_the_data(self, tmp_path) -> None:
        original = generate_dataset(
            seed=3, n_hotels=2, history_days=45, end_date=END_DATE
        )
        original.to_csv(tmp_path)
        restored = SyntheticDataset.from_csv(tmp_path)
        for name, frame in original.frames().items():
            pd.testing.assert_frame_equal(
                frame, restored.frames()[name], check_dtype=False
            )

    def test_missing_csv_directory_raises_an_actionable_error(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="generate_data"):
            SyntheticDataset.from_csv(tmp_path / "absent")

    @pytest.mark.parametrize("count", [0, len(HOTEL_CATALOG) + 1])
    def test_invalid_hotel_count_rejected(self, count: int) -> None:
        with pytest.raises(ValueError, match="n_hotels"):
            SyntheticDatasetGenerator(n_hotels=count, history_days=30)


# --------------------------------------------------------------------------- #
# Realism -- the relationships the models are supposed to learn
# --------------------------------------------------------------------------- #


class TestRealism:
    def test_business_hotels_empty_at_the_weekend(self, nights: pd.DataFrame) -> None:
        business = nights[nights.segment == MarketSegment.BUSINESS.value]
        weekday = business[~business.is_weekend].occupancy.mean()
        weekend = business[business.is_weekend].occupancy.mean()
        assert weekday > weekend + 0.15

    def test_leisure_hotels_fill_at_the_weekend(self, nights: pd.DataFrame) -> None:
        """The mirror image of the previous test. Both signs must exist, or a
        model learns a single global weekend effect that is wrong half the time."""
        leisure = nights[nights.segment == MarketSegment.LEISURE.value]
        weekday = leisure[~leisure.is_weekend].occupancy.mean()
        weekend = leisure[leisure.is_weekend].occupancy.mean()
        assert weekend > weekday + 0.08

    def test_goa_collapses_in_the_monsoon(self, nights: pd.DataFrame) -> None:
        goa = nights[nights.city == "Goa"]
        monsoon = goa[goa.month.isin([6, 7, 8, 9])].occupancy.mean()
        winter = goa[goa.month.isin([12, 1, 2])].occupancy.mean()
        assert winter > monsoon * 1.6

    def test_events_produce_demand_spikes(self, nights: pd.DataFrame) -> None:
        quiet = nights[nights.local_event_score < 0.1].occupancy.mean()
        busy = nights[nights.local_event_score > 0.5].occupancy.mean()
        assert busy > quiet + 0.10

    def test_occupancy_drives_achieved_rate(self, nights: pd.DataFrame) -> None:
        correlations = _within_group_corr(nights, "occupancy", "adr")
        assert correlations.min() > 0.40
        assert correlations.median() > 0.70

    def test_competitor_rates_track_the_same_demand(self, nights: pd.DataFrame) -> None:
        """Correlated, deliberately not identical -- competitors read the market
        through their own noise, which is what makes the feature informative
        rather than a copy of the label."""
        correlations = _within_group_corr(nights, "occupancy", "competitor_rate")
        assert correlations.min() > 0.30
        assert correlations.max() < 0.97

    def test_search_demand_leads_occupancy(self, nights: pd.DataFrame) -> None:
        correlations = _within_group_corr(nights, "occupancy", "search_demand")
        assert correlations.median() > 0.50

    def test_cancellation_risk_grows_with_lead_time(
        self, dataset: SyntheticDataset
    ) -> None:
        bookings = dataset.bookings
        buckets = pd.cut(bookings.lead_time_days, [-1, 7, 30, 60, MAX_LEAD_DAYS])
        rates = bookings.groupby(buckets, observed=True).apply(
            lambda d: d.cancellation_count.sum() / d.booking_count.sum()
        )
        assert rates.is_monotonic_increasing
        assert 0.03 < rates.iloc[0] < 0.15
        assert 0.15 < rates.iloc[-1] < 0.30

    def test_business_travel_books_later_than_leisure(
        self, dataset: SyntheticDataset
    ) -> None:
        merged = dataset.bookings.merge(
            dataset.hotels[["hotel_id", "segment"]], on="hotel_id"
        )
        weighted = merged.groupby("segment").apply(
            lambda d: np.average(d.lead_time_days, weights=d.booking_count)
        )
        assert weighted[MarketSegment.BUSINESS.value] < weighted[MarketSegment.LEISURE.value]

    def test_peak_dates_are_booked_further_ahead(self, dataset: SyntheticDataset) -> None:
        """High-demand nights sell out early, so the booking curve stretches."""
        merged = dataset.bookings.merge(
            dataset.demand_signals[["hotel_id", "room_type", "stay_date",
                                    "local_event_score"]],
            left_on=["hotel_id", "room_type", "check_in_date"],
            right_on=["hotel_id", "room_type", "stay_date"],
        )
        quiet = merged[merged.local_event_score < 0.1]
        busy = merged[merged.local_event_score > 0.5]
        assert np.average(busy.lead_time_days, weights=busy.booking_count) > np.average(
            quiet.lead_time_days, weights=quiet.booking_count
        )

    def test_competitor_coverage_is_incomplete(self, dataset: SyntheticDataset) -> None:
        """Missing competitor data is a case the pipeline must survive, so the
        generator produces it on purpose."""
        observed = len(dataset.competitor_prices)
        possible = (
            len(dataset.rooms) * 365 * len(Competitor) * len(COMPETITOR_OBSERVATION_LEADS)
        )
        coverage = observed / possible
        assert 0.7 < coverage < 0.95

    def test_competitor_set_has_a_real_price_spread(self, nights: pd.DataFrame) -> None:
        """``competitor_min_rate`` and ``competitor_max_rate`` must differ, or
        both features are the same column under two names."""
        with_rates = nights.dropna(subset=["competitor_min", "competitor_max"])
        spread = (
            with_rates.competitor_max - with_rates.competitor_min
        ) / with_rates.competitor_rate
        assert spread.median() > 0.05

    def test_demand_has_a_yearly_and_a_weekly_cycle(self, nights: pd.DataFrame) -> None:
        """Prophet is asked to fit exactly these two seasonalities in Phase 5."""
        goa = nights[(nights.city == "Goa") & (nights.room_type == "standard")]
        monthly = goa.groupby("month").occupancy.mean()
        assert monthly.max() - monthly.min() > 0.20

        business = nights[nights.segment == MarketSegment.BUSINESS.value]
        by_dow = business.groupby("dow").occupancy.mean()
        assert by_dow.max() - by_dow.min() > 0.15

    def test_there_is_a_trend_component(self, dataset: SyntheticDataset) -> None:
        """A slow drift the forecaster should pick up rather than treat as noise."""
        bookings = dataset.bookings.assign(
            sold=lambda d: d.booking_count - d.cancellation_count
        )
        daily = bookings.groupby("check_in_date").sold.sum()
        first_quarter = daily.iloc[:90].mean()
        last_quarter = daily.iloc[-90:].mean()
        assert first_quarter != pytest.approx(last_quarter, rel=0.01)


class TestSummary:
    def test_summary_reports_one_row_per_hotel(self, dataset: SyntheticDataset) -> None:
        summary = summarise(dataset)
        assert len(summary) == N_HOTELS
        assert {"rooms_sold", "cancellation_rate", "adr"} <= set(summary.columns)

    def test_summary_numbers_are_plausible(self, dataset: SyntheticDataset) -> None:
        summary = summarise(dataset)
        assert summary.cancellation_rate.between(0.05, 0.15).all()
        assert summary.adr.between(2_000, 30_000).all()
        assert summary.mean_lead_time.between(10, 45).all()

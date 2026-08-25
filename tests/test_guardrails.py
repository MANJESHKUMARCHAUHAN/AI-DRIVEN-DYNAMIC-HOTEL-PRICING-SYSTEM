"""Tests for the business guardrails.

Guardrails are the difference between a pricing model and a pricing *system*.
They are the last thing between a statistical output and a number a guest is
charged, so they get their own file and are tested harder than anything else in
the project.

Three groups:

* **The type gate.** :class:`FinalPrice` must be unconstructable outside
  :func:`apply`. This is what makes the guardrails structurally unbypassable
  rather than merely conventional.
* **Each rule in isolation**, including the boundary where it stops firing.
* **Ordering and interaction**, because "relative rules first, absolute rules
  last" is a correctness property: a floor that a relative rule can undercut is
  not a floor.
"""

from __future__ import annotations

import math

import pytest

from pricing.guardrails import (
    SANITY_MULTIPLE,
    FinalPrice,
    GuardrailContext,
    GuardrailHit,
    RawPrice,
    Rule,
    apply,
    describe,
)

BASE = 5_000.0


def raw(amount: float, base: float = BASE) -> RawPrice:
    return RawPrice(amount=amount, base_price=base, total_adjustment=0.0)


def ctx(**overrides) -> GuardrailContext:
    row = dict(base_price=BASE)
    row.update(overrides)
    return GuardrailContext(**row)


# --------------------------------------------------------------------------- #
# The type gate
# --------------------------------------------------------------------------- #


class TestTypeGate:
    def test_final_price_cannot_be_constructed_directly(self) -> None:
        """The load-bearing design decision of the whole pricing layer."""
        with pytest.raises(TypeError, match="only be created by guardrails.apply"):
            FinalPrice(amount=1.0, raw=raw(1.0), applied=[])

    def test_a_forged_token_does_not_work(self) -> None:
        with pytest.raises(TypeError):
            FinalPrice(amount=1.0, raw=raw(1.0), applied=[], _token=object())

    def test_apply_is_the_only_producer(self, settings) -> None:
        result = apply(raw(6_000.0), ctx(), settings)
        assert isinstance(result, FinalPrice)

    def test_final_price_carries_the_raw_price(self, settings) -> None:
        """A decision must always be able to show what it started from."""
        result = apply(raw(6_000.0), ctx(), settings)
        assert result.raw.amount == 6_000.0


# --------------------------------------------------------------------------- #
# Absolute limits
# --------------------------------------------------------------------------- #


class TestAbsoluteLimits:
    def test_price_below_the_floor_is_raised(self, settings) -> None:
        result = apply(raw(1_000.0), ctx(), settings)
        assert result.amount == settings.pricing.min_price
        assert Rule.MIN_PRICE.value in result.rules_applied

    def test_price_above_the_ceiling_is_capped(self, settings) -> None:
        result = apply(raw(90_000.0, base=80_000.0), ctx(base_price=80_000.0), settings)
        assert result.amount == settings.pricing.max_price
        assert Rule.MAX_PRICE.value in result.rules_applied

    def test_price_inside_the_limits_is_untouched(self, settings) -> None:
        result = apply(raw(6_000.0), ctx(), settings)
        assert result.amount == 6_000.0
        assert result.applied == []

    @pytest.mark.parametrize("amount", [2_500.0, 25_000.0])
    def test_exactly_on_a_limit_does_not_fire(self, settings, amount: float) -> None:
        """Boundaries are inclusive; a rule that fires at exactly the limit
        would make the limit unreachable."""
        result = apply(raw(amount, base=amount), ctx(base_price=amount), settings)
        assert result.amount == amount
        assert result.applied == []

    def test_room_ceiling_applies_inside_the_global_one(self, settings) -> None:
        result = apply(raw(9_000.0), ctx(room_ceiling_price=7_500.0), settings)
        assert result.amount == 7_500.0
        assert Rule.ROOM_CEILING.value in result.rules_applied

    def test_room_floor_applies_inside_the_global_one(self, settings) -> None:
        result = apply(raw(3_000.0), ctx(room_floor_price=4_000.0), settings)
        assert result.amount == 4_000.0
        assert Rule.ROOM_FLOOR.value in result.rules_applied


# --------------------------------------------------------------------------- #
# Sanity
# --------------------------------------------------------------------------- #


class TestSanity:
    @pytest.mark.parametrize("amount", [float("nan"), float("inf"), -100.0, 0.0])
    def test_unusable_model_output_falls_back_to_base(self, settings, amount) -> None:
        """A broken model is not an aggressive one. Knowably safe beats
        confidently wrong."""
        result = apply(raw(amount), ctx(), settings)
        assert math.isfinite(result.amount)
        assert result.amount == BASE
        assert Rule.SANITY.value in result.rules_applied

    def test_absurd_multiple_of_base_is_cut_back(self, settings) -> None:
        result = apply(raw(BASE * 20), ctx(), settings)
        assert Rule.SANITY.value in result.rules_applied
        assert result.amount <= BASE * SANITY_MULTIPLE

    def test_a_merely_aggressive_price_is_not_a_sanity_failure(self, settings) -> None:
        result = apply(raw(BASE * 1.4), ctx(), settings)
        assert Rule.SANITY.value not in result.rules_applied


# --------------------------------------------------------------------------- #
# Daily change cap
# --------------------------------------------------------------------------- #


class TestDailyChangeCap:
    def test_large_rise_is_capped(self, settings) -> None:
        """A rate that jumps overnight reads as a pricing error whether or not
        it is one."""
        result = apply(raw(9_000.0), ctx(current_price=6_000.0), settings)
        assert result.amount == pytest.approx(6_000.0 * 1.15)
        assert Rule.MAX_DAILY_RISE.value in result.rules_applied

    def test_large_fall_is_capped(self, settings) -> None:
        result = apply(raw(3_000.0), ctx(current_price=6_000.0), settings)
        assert result.amount == pytest.approx(6_000.0 * 0.85)
        assert Rule.MAX_DAILY_FALL.value in result.rules_applied

    def test_a_move_inside_the_cap_is_allowed(self, settings) -> None:
        result = apply(raw(6_500.0), ctx(current_price=6_000.0), settings)
        assert result.amount == 6_500.0
        assert result.applied == []

    def test_exactly_at_the_cap_does_not_fire(self, settings) -> None:
        result = apply(raw(6_900.0), ctx(current_price=6_000.0), settings)
        assert result.applied == []

    def test_without_a_current_price_the_cap_is_skipped(self, settings) -> None:
        """No previous price means no day-over-day change to cap. Skipped
        rather than guessed."""
        result = apply(raw(9_000.0), ctx(current_price=None), settings)
        assert Rule.MAX_DAILY_RISE.value not in result.rules_applied

    def test_zero_current_price_does_not_divide_by_zero(self, settings) -> None:
        result = apply(raw(6_000.0), ctx(current_price=0.0), settings)
        assert result.amount == 6_000.0


# --------------------------------------------------------------------------- #
# Competitor bands
# --------------------------------------------------------------------------- #


class TestCompetitorBands:
    def test_far_above_the_market_is_capped(self, settings) -> None:
        result = apply(raw(12_000.0), ctx(competitor_max_rate=7_000.0), settings)
        assert result.amount == pytest.approx(7_000.0 * 1.20)
        assert Rule.COMPETITOR_UPPER.value in result.rules_applied

    def test_far_below_the_market_is_held_up(self, settings) -> None:
        """Undercutting the whole market buys volume the hotel cannot service."""
        result = apply(raw(3_000.0), ctx(competitor_min_rate=6_000.0), settings)
        assert result.amount == pytest.approx(6_000.0 * 0.80)
        assert Rule.COMPETITOR_LOWER.value in result.rules_applied

    def test_inside_the_band_is_untouched(self, settings) -> None:
        result = apply(
            raw(7_000.0),
            ctx(competitor_min_rate=6_000.0, competitor_max_rate=7_500.0),
            settings,
        )
        assert result.applied == []

    def test_exceptional_demand_lifts_the_upper_band(self, settings) -> None:
        """A city that is genuinely selling out is exactly when a hotel should
        be allowed to price above the market."""
        result = apply(
            raw(12_000.0),
            ctx(competitor_max_rate=7_000.0, allow_increase_override=True),
            settings,
        )
        assert Rule.COMPETITOR_UPPER.value not in result.rules_applied

    def test_the_override_does_not_defeat_the_absolute_ceiling(self, settings) -> None:
        """Exceptional demand is not a licence to leave the absolute limits."""
        result = apply(
            raw(99_000.0, base=30_000.0),
            ctx(base_price=30_000.0, competitor_max_rate=7_000.0,
                allow_increase_override=True),
            settings,
        )
        assert result.amount <= settings.pricing.max_price

    def test_missing_competitor_data_disables_the_bands(self, settings) -> None:
        """Pricing against an imagined market is worse than pricing against
        none."""
        result = apply(raw(12_000.0), ctx(), settings)
        assert Rule.COMPETITOR_UPPER.value not in result.rules_applied
        assert Rule.COMPETITOR_LOWER.value not in result.rules_applied


# --------------------------------------------------------------------------- #
# Low occupancy
# --------------------------------------------------------------------------- #


class TestLowOccupancyBlock:
    def test_no_increase_when_the_hotel_is_not_selling(self, settings) -> None:
        result = apply(
            raw(7_000.0), ctx(current_price=6_000.0, occupancy_rate=0.20), settings
        )
        assert result.amount <= 6_000.0
        assert Rule.LOW_OCCUPANCY_BLOCK.value in result.rules_applied

    def test_decreases_are_still_allowed_at_low_occupancy(self, settings) -> None:
        """The rule blocks rises, not the discounting that low occupancy calls
        for."""
        result = apply(
            raw(5_500.0), ctx(current_price=6_000.0, occupancy_rate=0.20), settings
        )
        assert result.amount == 5_500.0
        assert Rule.LOW_OCCUPANCY_BLOCK.value not in result.rules_applied

    def test_high_occupancy_permits_a_rise(self, settings) -> None:
        result = apply(
            raw(6_500.0), ctx(current_price=6_000.0, occupancy_rate=0.85), settings
        )
        assert result.amount == 6_500.0

    def test_exactly_at_the_threshold_permits_a_rise(self, settings) -> None:
        result = apply(
            raw(6_400.0),
            ctx(current_price=6_000.0,
                occupancy_rate=settings.pricing.low_occupancy_threshold),
            settings,
        )
        assert Rule.LOW_OCCUPANCY_BLOCK.value not in result.rules_applied

    def test_unknown_occupancy_does_not_block(self, settings) -> None:
        result = apply(
            raw(6_500.0), ctx(current_price=6_000.0, occupancy_rate=None), settings
        )
        assert Rule.LOW_OCCUPANCY_BLOCK.value not in result.rules_applied


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


class TestOrdering:
    def test_absolute_floor_wins_over_the_competitor_band(self, settings) -> None:
        """The correctness property: a floor a relative rule can undercut is
        not a floor."""
        result = apply(raw(500.0), ctx(competitor_min_rate=2_000.0), settings)
        assert result.amount >= settings.pricing.min_price

    def test_absolute_ceiling_wins_over_the_daily_cap(self, settings) -> None:
        result = apply(
            raw(99_000.0, base=30_000.0),
            ctx(base_price=30_000.0, current_price=30_000.0),
            settings,
        )
        assert result.amount <= settings.pricing.max_price

    def test_relative_rules_are_recorded_before_absolute_ones(self, settings) -> None:
        result = apply(
            raw(50_000.0, base=20_000.0),
            ctx(base_price=20_000.0, current_price=20_000.0, competitor_max_rate=8_000.0),
            settings,
        )
        rules = result.rules_applied
        relative = {Rule.MAX_DAILY_RISE.value, Rule.COMPETITOR_UPPER.value}
        absolute = {Rule.MIN_PRICE.value, Rule.MAX_PRICE.value}
        indexes_relative = [i for i, r in enumerate(rules) if r in relative]
        indexes_absolute = [i for i, r in enumerate(rules) if r in absolute]
        if indexes_relative and indexes_absolute:
            assert max(indexes_relative) < min(indexes_absolute)

    def test_every_hit_records_before_and_after(self, settings) -> None:
        result = apply(raw(500.0), ctx(current_price=6_000.0), settings)
        for hit in result.applied:
            assert hit.before != hit.after
            assert hit.reason

    def test_the_result_always_sits_inside_the_absolute_limits(self, settings) -> None:
        """The single invariant that must hold no matter what comes in."""
        limits = settings.pricing
        for amount in (-500.0, 0.0, 1.0, 4_999.0, 50_000.0, 1e9, float("nan")):
            for current in (None, 100.0, 6_000.0, 40_000.0):
                result = apply(raw(amount), ctx(current_price=current), settings)
                assert limits.min_price <= result.amount <= limits.max_price, (
                    amount,
                    current,
                )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


class TestReporting:
    def test_no_guardrails_reads_cleanly(self, settings) -> None:
        result = apply(raw(6_000.0), ctx(), settings)
        assert result.was_clamped is False
        assert describe(result) == ["No guardrails were triggered."]

    def test_a_fired_rule_is_described_with_both_values(self, settings) -> None:
        result = apply(raw(1_000.0), ctx(), settings)
        text = describe(result)[0]
        assert "MIN_PRICE_FLOOR" in text
        assert "1,000" in text and "2,500" in text

    def test_as_dict_is_api_ready(self, settings) -> None:
        result = apply(raw(1_000.0), ctx(), settings)
        payload = result.as_dict()
        assert payload["final_price"] == settings.pricing.min_price
        assert payload["raw_price"] == 1_000.0
        assert payload["guardrails_applied"][0]["rule"] == Rule.MIN_PRICE.value

    def test_hit_delta_is_signed(self) -> None:
        hit = GuardrailHit(Rule.MIN_PRICE, before=1_000.0, after=2_500.0, reason="x")
        assert hit.delta == 1_500.0

    def test_rounding_follows_configuration(self, settings) -> None:
        """PRICE_ROUNDING=0 means whole rupees; nobody quotes a rate in paise."""
        result = apply(raw(6_123.456), ctx(), settings)
        assert result.amount == round(result.amount)

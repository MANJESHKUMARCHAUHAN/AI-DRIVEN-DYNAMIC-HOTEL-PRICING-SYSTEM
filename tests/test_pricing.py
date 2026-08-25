"""Tests for the pricing rules, the demand engine and the pricing engine.

The rules are pure arithmetic, so most of these are one-liners. The ones worth
reading are the *sign* tests: they pin down the commercial judgement rather than
the formula. "Discount a half-empty hotel on the day, but hold the price if
there are still three weeks to sell" is a business decision, and it is the kind
of thing a well-meaning refactor silently inverts.

The guardrails have their own file; here they are only checked where they
interact with the engine.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd
import pytest

from database.models import RoomType, Season
from pricing.demand_engine import (
    FALLBACK_CONFIDENCE,
    SINGLE_MODEL_CONFIDENCE_CAP,
    DemandEngine,
    DemandEstimate,
)
from pricing.guardrails import Rule
from pricing.pricing_engine import (
    EXCEPTIONAL_DEMAND,
    PriceDecision,
    PricingEngine,
    PricingRequest,
)
from pricing.rules import (
    HIGH_OCCUPANCY,
    LOW_OCCUPANCY,
    Adjustment,
    competitor_adjustment,
    demand_adjustment,
    event_adjustment,
    occupancy_adjustment,
    season_adjustment,
)

STAY = date(2026, 9, 15)
BASE = 5_000.0


def estimate(**overrides) -> DemandEstimate:
    row = dict(
        blended=0.70,
        prophet=0.72,
        gbr=0.68,
        weight=0.5,
        confidence=0.85,
        lower=0.62,
        upper=0.78,
        sources=["prophet", "gradient_boosting"],
    )
    row.update(overrides)
    return DemandEstimate(**row)


def request(**overrides) -> PricingRequest:
    row = dict(
        hotel_id="H001",
        room_type=RoomType.DELUXE,
        check_in_date=STAY,
        base_price=BASE,
        days_to_checkin=14,
    )
    row.update(overrides)
    return PricingRequest(**row)


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #


class TestDemandRule:
    def test_above_baseline_raises_the_price(self) -> None:
        assert demand_adjustment(0.85, baseline_demand=0.65).value > 0

    def test_below_baseline_lowers_it(self) -> None:
        assert demand_adjustment(0.45, baseline_demand=0.65).value < 0

    def test_at_baseline_is_neutral(self) -> None:
        assert demand_adjustment(0.65, baseline_demand=0.65).value == pytest.approx(0.0)

    def test_it_is_the_ratio_not_the_level(self) -> None:
        """70% is strong for a hotel that runs at 55% and weak for one at 85%.
        One rule has to work for the whole estate."""
        strong = demand_adjustment(0.70, baseline_demand=0.55).value
        weak = demand_adjustment(0.70, baseline_demand=0.85).value
        assert strong > 0 > weak

    def test_sensitivity_damps_forecast_noise(self) -> None:
        """Demand is forecast with a real error bar; passing it through at 1:1
        turns forecast noise into rate volatility."""
        deviation = 0.85 / 0.65 - 1.0
        assert demand_adjustment(0.85, baseline_demand=0.65).raw_value < deviation

    def test_extreme_demand_is_clamped(self) -> None:
        adjustment = demand_adjustment(2.0, baseline_demand=0.65, limit=0.25)
        assert adjustment.value == 0.25
        assert adjustment.clamped is True

    @pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
    def test_unusable_input_falls_back_to_neutral(self, bad) -> None:
        """A NaN reaching the multiplication turns the whole price into NaN."""
        adjustment = demand_adjustment(bad, baseline_demand=0.65)
        assert math.isfinite(adjustment.value)
        assert adjustment.value == pytest.approx(0.0)


class TestOccupancyRule:
    """The occupancy x lead-time interaction, corner by corner."""

    def test_high_occupancy_far_out_raises_hard(self) -> None:
        assert occupancy_adjustment(0.90, 30).value > 0.10

    def test_high_occupancy_near_raises_only_modestly(self) -> None:
        """Nearly sold out anyway -- there is little left for a rise to earn on."""
        far = occupancy_adjustment(0.90, 30).value
        near = occupancy_adjustment(0.90, 1).value
        assert 0 < near < far

    def test_low_occupancy_far_out_holds(self) -> None:
        """There is still time to sell; discounting now just gives away margin."""
        assert occupancy_adjustment(0.15, 30).value == pytest.approx(0.0, abs=0.01)

    def test_low_occupancy_near_discounts(self) -> None:
        """An unsold room-night is worth exactly zero at midnight."""
        assert occupancy_adjustment(0.15, 1).value < -0.05

    def test_on_pace_is_neutral(self) -> None:
        assert occupancy_adjustment(0.60, 14).value == pytest.approx(0.0)

    def test_the_discount_deepens_as_time_runs_out(self) -> None:
        values = [occupancy_adjustment(0.15, lead).value for lead in (21, 14, 7, 0)]
        assert values == sorted(values, reverse=True)
        assert values[-1] < values[0]

    def test_thresholds_are_the_documented_ones(self) -> None:
        assert occupancy_adjustment(HIGH_OCCUPANCY, 30).value == pytest.approx(0.0)
        assert occupancy_adjustment(LOW_OCCUPANCY, 0).value == pytest.approx(0.0)

    def test_reason_explains_the_corner(self) -> None:
        assert "discount" in occupancy_adjustment(0.10, 1).reason
        assert "still time" in occupancy_adjustment(0.10, 30).reason


class TestCompetitorRule:
    def test_market_above_us_creates_headroom(self) -> None:
        assert competitor_adjustment(BASE, 6_500.0).value > 0

    def test_market_below_us_pulls_us_down(self) -> None:
        assert competitor_adjustment(BASE, 4_000.0).value < 0

    def test_level_with_the_market_is_neutral(self) -> None:
        assert competitor_adjustment(BASE, BASE).value == pytest.approx(0.0)

    def test_we_move_only_part_of_the_way(self) -> None:
        """Following the market one-for-one is how two automated systems talk
        each other into a race to the bottom."""
        gap = (6_500.0 - BASE) / BASE
        assert 0 < competitor_adjustment(BASE, 6_500.0).raw_value < gap

    def test_missing_competitor_data_contributes_nothing(self) -> None:
        """Pricing against an imagined market is worse than pricing against
        none."""
        adjustment = competitor_adjustment(BASE, None, competitor_missing=True)
        assert adjustment.value == 0.0
        assert "no competitor rate" in adjustment.reason

    def test_an_extreme_competitor_rate_is_clamped(self) -> None:
        assert competitor_adjustment(BASE, 50_000.0, limit=0.15).value == 0.15

    @pytest.mark.parametrize("bad", [0.0, -100.0])
    def test_unusable_competitor_rate_is_ignored(self, bad: float) -> None:
        assert competitor_adjustment(BASE, bad).value == 0.0


class TestSeasonRule:
    def test_monsoon_weakens_rates(self) -> None:
        assert season_adjustment(Season.MONSOON).value < 0

    def test_winter_supports_them(self) -> None:
        assert season_adjustment(Season.WINTER).value > 0

    def test_it_stays_small(self) -> None:
        """The base rate already has the season in it; a large term here would
        double-count."""
        for season in Season:
            assert abs(season_adjustment(season).value) <= 0.10

    def test_absent_season_is_neutral(self) -> None:
        assert season_adjustment(None).value == 0.0


class TestEventRule:
    def test_nothing_happening_is_neutral(self) -> None:
        assert event_adjustment(0.0).value == 0.0

    def test_an_event_raises_the_price(self) -> None:
        assert event_adjustment(0.8).value > 0

    def test_the_weekend_alone_raises_it_a_little(self) -> None:
        assert 0 < event_adjustment(0.0, is_weekend=True).value < event_adjustment(0.8).value

    def test_pressures_combine_with_diminishing_returns(self) -> None:
        """A festival on a bank holiday weekend is busier than any one of them,
        but the city cannot be more than full."""
        festival = event_adjustment(0.8).value
        everything = event_adjustment(0.8, is_weekend=True, is_holiday=True).value
        assert festival < everything <= 0.15

    def test_it_is_bounded_by_the_limit(self) -> None:
        assert event_adjustment(1.0, is_weekend=True, is_holiday=True, limit=0.15).value <= 0.15


class TestAdjustmentType:
    def test_clamped_is_derived_not_asserted(self) -> None:
        assert Adjustment("x", 0.25, 0.40, "r", {}).clamped is True
        assert Adjustment("x", 0.25, 0.25, "r", {}).clamped is False

    def test_describe_is_one_readable_line(self) -> None:
        text = Adjustment("demand", 0.12, 0.12, "because", {}).describe()
        assert "Demand" in text and "12.0%" in text


# --------------------------------------------------------------------------- #
# Demand engine
# --------------------------------------------------------------------------- #


class FakeProphet:
    def __init__(self, value=0.80, fail=False):
        self.value, self.fail = value, fail

    def demand_on(self, hotel_id, room_type, day):
        if self.fail:
            raise RuntimeError("stan exploded")
        if self.value is None:
            return None
        return {
            "forecast": self.value,
            "lower": self.value - 0.06,
            "upper": self.value + 0.06,
            "trend": self.value,
        }


class FakeGBR:
    model = object()  # the engine checks this to decide availability

    def __init__(self, value=0.70, fail=False):
        self.value, self.fail = value, fail

    def predict_one(self, frame):
        if self.fail:
            raise RuntimeError("feature mismatch")
        return {
            "demand": self.value,
            "lower": self.value - 0.08,
            "upper": self.value + 0.08,
            "confidence": 0.8,
        }


FEATURES = pd.DataFrame([{"occupancy_rate": 0.7}])


class TestDemandEngine:
    def test_blends_both_models_at_the_configured_weight(self, settings) -> None:
        engine = DemandEngine(
            prophet_bundle=FakeProphet(0.80), gbr_model=FakeGBR(0.60), settings=settings
        )
        result = engine.estimate("H001", RoomType.DELUXE, STAY, FEATURES)
        expected = settings.model.model_prophet_blend_weight * 0.80 + (
            1 - settings.model.model_prophet_blend_weight
        ) * 0.60
        assert result.blended == pytest.approx(expected)
        assert result.sources == ["prophet", "gradient_boosting"]
        assert result.degraded is False

    def test_missing_prophet_collapses_onto_the_gbr(self, settings) -> None:
        engine = DemandEngine(gbr_model=FakeGBR(0.66), settings=settings)
        result = engine.estimate("H001", RoomType.DELUXE, STAY, FEATURES)
        assert result.blended == pytest.approx(0.66)
        assert result.sources == ["gradient_boosting"]
        assert result.degraded is True
        assert result.confidence <= SINGLE_MODEL_CONFIDENCE_CAP

    def test_missing_gbr_collapses_onto_prophet(self, settings) -> None:
        engine = DemandEngine(prophet_bundle=FakeProphet(0.75), settings=settings)
        result = engine.estimate("H001", RoomType.DELUXE, STAY, FEATURES)
        assert result.blended == pytest.approx(0.75)
        assert result.sources == ["prophet"]

    def test_a_throwing_model_is_treated_as_absent(self, settings) -> None:
        """A pricing API that 500s because a model file is corrupt has turned a
        degraded feature into an outage."""
        engine = DemandEngine(
            prophet_bundle=FakeProphet(fail=True),
            gbr_model=FakeGBR(0.70),
            settings=settings,
        )
        result = engine.estimate("H001", RoomType.DELUXE, STAY, FEATURES)
        assert result.blended == pytest.approx(0.70)
        assert result.degraded is True

    def test_no_models_falls_back_to_history(self, settings) -> None:
        engine = DemandEngine(settings=settings)
        result = engine.estimate(
            "H001", RoomType.DELUXE, STAY, FEATURES, fallback_demand=0.58
        )
        assert result.blended == pytest.approx(0.58)
        assert result.confidence == FALLBACK_CONFIDENCE
        assert "historical" in result.notes[0]

    def test_no_models_and_no_history_uses_the_baseline(self, settings) -> None:
        result = DemandEngine(settings=settings).estimate(
            "H001", RoomType.DELUXE, STAY, FEATURES
        )
        assert result.blended == pytest.approx(settings.pricing.baseline_demand)
        assert result.degraded is True

    def test_agreement_earns_more_confidence_than_disagreement(self, settings) -> None:
        """Two confident models that disagree is exactly when the blend is
        least trustworthy, and averaging them hides it."""
        agree = DemandEngine(
            prophet_bundle=FakeProphet(0.70), gbr_model=FakeGBR(0.71), settings=settings
        ).estimate("H001", RoomType.DELUXE, STAY, FEATURES)
        disagree = DemandEngine(
            prophet_bundle=FakeProphet(0.35), gbr_model=FakeGBR(0.95), settings=settings
        ).estimate("H001", RoomType.DELUXE, STAY, FEATURES)
        assert agree.confidence > disagree.confidence

    def test_large_disagreement_is_noted(self, settings) -> None:
        result = DemandEngine(
            prophet_bundle=FakeProphet(0.30), gbr_model=FakeGBR(0.90), settings=settings
        ).estimate("H001", RoomType.DELUXE, STAY, FEATURES)
        assert result.disagreement == pytest.approx(0.60)
        assert any("disagree" in note for note in result.notes)

    def test_available_models_is_reported(self, settings) -> None:
        engine = DemandEngine(prophet_bundle=FakeProphet(), settings=settings)
        assert engine.available_models == ["prophet"]

    def test_confidence_is_always_a_fraction(self, settings) -> None:
        for prophet_value, gbr_value in ((0.1, 0.9), (0.5, 0.5), (1.2, 0.0)):
            result = DemandEngine(
                prophet_bundle=FakeProphet(prophet_value),
                gbr_model=FakeGBR(gbr_value),
                settings=settings,
            ).estimate("H001", RoomType.DELUXE, STAY, FEATURES)
            assert 0.0 <= result.confidence <= 1.0


# --------------------------------------------------------------------------- #
# Pricing engine
# --------------------------------------------------------------------------- #


class TestPricingEngine:
    def test_all_five_adjustments_are_computed(self, settings) -> None:
        decision = PricingEngine(settings).price(request(), estimate())
        assert [a.name for a in decision.adjustments] == [
            "demand", "occupancy", "competitor", "season", "event"
        ]

    def test_adjustments_are_summed_not_chained(self, settings) -> None:
        """Additive terms are what a revenue manager can check by hand."""
        decision = PricingEngine(settings).price(
            request(occupancy_rate=0.60, competitor_rate=BASE, season=None),
            estimate(blended=0.65, confidence=1.0),
        )
        assert decision.total_adjustment == pytest.approx(
            sum(a.value for a in decision.adjustments)
        )

    def test_raw_price_is_base_times_one_plus_total(self, settings) -> None:
        decision = PricingEngine(settings).price(request(), estimate(confidence=1.0))
        assert decision.raw_price == pytest.approx(
            BASE * (1 + decision.total_adjustment)
        )

    def test_strong_demand_raises_the_price(self, settings) -> None:
        engine = PricingEngine(settings)
        weak = engine.price(request(), estimate(blended=0.35))
        strong = engine.price(request(), estimate(blended=0.95))
        assert strong.final_price > weak.final_price

    def test_low_confidence_moves_the_price_less(self, settings) -> None:
        """'The models are unsure' should degrade towards the base rate, not
        towards an arbitrary number."""
        engine = PricingEngine(settings)
        sure = engine.price(request(), estimate(blended=0.95, confidence=1.0))
        unsure = engine.price(request(), estimate(blended=0.95, confidence=0.1))
        assert abs(sure.raw_price - BASE) > abs(unsure.raw_price - BASE)

    def test_a_zero_base_price_is_rejected(self, settings) -> None:
        with pytest.raises(ValueError, match="base_price must be positive"):
            PricingEngine(settings).price(request(base_price=0.0), estimate())

    def test_lead_time_is_derived_when_not_supplied(self, settings) -> None:
        far = request(days_to_checkin=None, check_in_date=date.today() + timedelta(days=40))
        assert far.resolved_days_to_checkin() == 40

    def test_a_past_date_gives_zero_lead_time(self) -> None:
        past = request(days_to_checkin=None, check_in_date=date.today() - timedelta(days=3))
        assert past.resolved_days_to_checkin() == 0

    def test_the_final_price_respects_the_absolute_limits(self, settings) -> None:
        engine = PricingEngine(settings)
        for demand_value in (0.0, 0.5, 1.0, 5.0):
            decision = engine.price(
                request(event_score=1.0, is_holiday=True, is_weekend=True),
                estimate(blended=demand_value, confidence=1.0),
            )
            assert settings.pricing.min_price <= decision.final_price <= settings.pricing.max_price

    def test_guardrails_are_recorded_on_the_decision(self, settings) -> None:
        """Strong demand on a hotel that is not selling: the rules say raise,
        the guardrail says no."""
        decision = PricingEngine(settings).price(
            request(current_price=4_500.0, occupancy_rate=0.10),
            estimate(blended=0.99, confidence=1.0),
        )
        assert decision.raw_price > 4_500.0, "the test needs a rise to block"
        assert Rule.LOW_OCCUPANCY_BLOCK.value in decision.guardrails_applied
        assert decision.final_price <= 4_500.0

    def test_exceptional_demand_escapes_the_competitor_band(self, settings) -> None:
        engine = PricingEngine(settings)
        capped = engine.price(
            request(competitor_max_rate=5_200.0, competitor_rate=5_200.0),
            estimate(blended=0.80, confidence=1.0),
        )
        uncapped = engine.price(
            request(competitor_max_rate=5_200.0, competitor_rate=5_200.0),
            estimate(blended=EXCEPTIONAL_DEMAND + 0.02, confidence=1.0),
        )
        assert uncapped.final_price >= capped.final_price

    def test_price_change_is_measured_against_the_current_price(self, settings) -> None:
        decision = PricingEngine(settings).price(
            request(current_price=6_000.0), estimate(blended=0.70)
        )
        assert decision.price_change_percent == pytest.approx(
            (decision.final_price - 6_000.0) / 6_000.0 * 100.0
        )

    def test_without_a_current_price_change_is_measured_against_base(
        self, settings
    ) -> None:
        decision = PricingEngine(settings).price(request(), estimate())
        assert decision.price_change_percent == pytest.approx(
            (decision.final_price - BASE) / BASE * 100.0
        )


class TestDecisionRecord:
    @pytest.fixture
    def decision(self, settings) -> PriceDecision:
        return PricingEngine(settings).price(
            request(
                current_price=6_000.0,
                occupancy_rate=0.72,
                competitor_rate=6_500.0,
                competitor_min_rate=6_100.0,
                competitor_max_rate=6_900.0,
                season=Season.MONSOON,
                event_score=0.55,
            ),
            estimate(blended=0.82, confidence=0.87),
        )

    def test_as_dict_carries_the_whole_calculation(self, decision) -> None:
        """The audit trail: inputs, every adjustment, raw, guardrails, final."""
        payload = decision.as_dict()
        assert payload["base_price"] == BASE
        assert len(payload["adjustments"]) == 5
        assert "raw_recommended_price" in payload
        assert "final_recommended_price" in payload
        assert payload["demand"]["confidence"] == 0.87

    def test_every_adjustment_carries_its_reasoning(self, decision) -> None:
        for adjustment in decision.as_dict()["adjustments"]:
            assert adjustment["reason"]
            assert "inputs" in adjustment

    def test_adjustment_lookup(self, decision) -> None:
        assert decision.adjustment("demand") is not None
        assert decision.adjustment("nonexistent") is None

    def test_adjustment_map_is_what_the_audit_row_stores(self, decision) -> None:
        mapping = decision.adjustment_map()
        assert set(mapping) == {"demand", "occupancy", "competitor", "season", "event"}

    def test_explain_reads_like_the_specification(self, decision) -> None:
        text = decision.explain()
        assert "Base price:" in text
        assert "Demand" in text and "Occupancy" in text and "Competitor" in text
        assert "Raw price:" in text
        assert "Final price:" in text
        assert "confidence" in text

    def test_explain_lines_up_negative_percentages(self, decision) -> None:
        """Monsoon is negative; the column must not shift because of the sign."""
        lines = [
            line for line in decision.explain().splitlines()
            if line.startswith("  Season") or line.startswith("  Demand")
        ]
        assert len(lines) == 2
        assert lines[0].index("%") == lines[1].index("%")

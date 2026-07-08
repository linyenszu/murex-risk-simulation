from __future__ import annotations

import math

import pandas as pd
import pytest

from src.pricing.yield_curve import (
    ZeroCouponYieldCurve,
    bootstrap_zero_coupon_curve,
    price_zero_coupon_from_curve,
    sample_bond_market_quotes,
    validate_bond_quotes,
)


def test_bootstrap_zero_coupon_curve_returns_sorted_positive_curve() -> None:
    curve = bootstrap_zero_coupon_curve(sample_bond_market_quotes())

    assert list(curve["MaturityYears"]) == sorted(curve["MaturityYears"])
    assert (curve["DiscountFactor"] > 0).all()
    assert curve["ZeroRate"].notna().all()


def test_zero_coupon_pillar_matches_closed_form_discount_factor() -> None:
    quotes = pd.DataFrame(
        {
            "InstrumentID": ["ZC_1Y"],
            "MaturityYears": [1.0],
            "CouponRate": [0.0],
            "MarketPrice": [95.0],
            "FaceValue": [100.0],
            "Frequency": [1],
        }
    )
    curve = bootstrap_zero_coupon_curve(quotes)
    assert curve.loc[0, "DiscountFactor"] == pytest.approx(0.95)
    assert curve.loc[0, "ZeroRate"] == pytest.approx(-math.log(0.95))


def test_curve_reprices_input_coupon_bonds_at_pillars() -> None:
    quotes = sample_bond_market_quotes()
    zero_curve = ZeroCouponYieldCurve.from_bond_quotes(quotes)

    for _, bond in validate_bond_quotes(quotes).iterrows():
        frequency = int(bond["Frequency"])
        maturity = float(bond["MaturityYears"])
        periods = max(int(round(maturity * frequency)), 1)
        coupon = float(bond["FaceValue"]) * float(bond["CouponRate"]) / frequency
        pv = 0.0
        for i in range(1, periods + 1):
            t = i / frequency
            cashflow = coupon + (float(bond["FaceValue"]) if i == periods else 0.0)
            pv += cashflow * zero_curve.discount_factor(t)
        assert pv == pytest.approx(float(bond["MarketPrice"]), abs=0.35)


def test_zero_curve_interpolation_and_forward_rate_are_positive() -> None:
    zero_curve = ZeroCouponYieldCurve.from_bond_quotes(sample_bond_market_quotes())
    interpolated_rate = zero_curve.zero_rate(4.0)
    forward_rate = zero_curve.forward_rate(1.0, 5.0)

    assert interpolated_rate > 0
    assert forward_rate > 0


def test_price_zero_coupon_from_curve_uses_discount_factor() -> None:
    zero_curve = ZeroCouponYieldCurve.from_bond_quotes(sample_bond_market_quotes())
    price = price_zero_coupon_from_curve(1_000_000.0, 5.0, zero_curve)
    assert price == pytest.approx(1_000_000.0 * zero_curve.discount_factor(5.0))


def test_validate_bond_quotes_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="Missing required"):
        validate_bond_quotes(pd.DataFrame({"MaturityYears": [1.0]}))

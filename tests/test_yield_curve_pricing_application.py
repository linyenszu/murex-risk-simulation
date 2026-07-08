from __future__ import annotations

import math

import pandas as pd
import pytest

from main import build_market_context, load_zero_curve, run
from src.config.settings import RiskSettings
from src.pricing.instruments import MarketContext, price_fixed_rate_bond, price_zero_coupon_bond
from src.pricing.yield_curve import ZeroCouponYieldCurve, sample_bond_market_quotes


def test_market_context_uses_zero_curve_discount_factor() -> None:
    curve = ZeroCouponYieldCurve.from_bond_quotes(sample_bond_market_quotes())
    ctx = MarketContext(
        valuation_date=pd.Timestamp("2025-04-04"),
        risk_free_rate=0.02,
        dividend_yield=0.0,
        vols={},
        fx_foreign_rates={},
        zero_curve=curve,
    )

    assert ctx.discount_factor(5.0) == pytest.approx(curve.discount_factor(5.0))
    assert ctx.zero_rate(5.0) == pytest.approx(curve.zero_rate(5.0))
    assert ctx.discount_factor(5.0) != pytest.approx(math.exp(-0.02 * 5.0))


def test_zero_coupon_bond_uses_curve_when_no_trade_yield_supplied() -> None:
    curve = ZeroCouponYieldCurve.from_bond_quotes(sample_bond_market_quotes())
    ctx = MarketContext(pd.Timestamp("2025-04-04"), 0.02, 0.0, {}, {}, zero_curve=curve)
    maturity = pd.Timestamp("2030-04-04")
    result = price_zero_coupon_bond(2.0, 1000.0, maturity, ctx, yield_rate=None)

    assert result["NPV"] == pytest.approx(2.0 * 1000.0 * curve.discount_factor(5.0), rel=1e-3)


def test_fixed_rate_bond_uses_curve_cashflow_discounting() -> None:
    curve = ZeroCouponYieldCurve.from_bond_quotes(sample_bond_market_quotes())
    ctx = MarketContext(pd.Timestamp("2025-04-04"), 0.02, 0.0, {}, {}, zero_curve=curve)
    maturity = pd.Timestamp("2027-04-04")
    result = price_fixed_rate_bond(1.0, 100.0, 0.035, maturity, ctx, yield_rate=None, frequency=2)

    # This bond is one of the input-like quote structures; using the curve should price near par quote.
    assert result["NPV"] == pytest.approx(99.1, abs=0.5)


def test_main_pipeline_returns_bootstrapped_curve_and_prices_rates_products() -> None:
    results = run(
        RiskSettings(lookback_days=60, stress_window_days=30, random_seed=123),
        option_pricing_engine="black-scholes",
        mc_paths=100,
        tree_steps=10,
        bond_quotes_path="data/raw/bond_market_quotes.csv",
        use_yield_curve=True,
    )

    zero_curve = results["zero_curve"]
    instrument_risk = results["instrument_risk"]

    assert not zero_curve.empty
    rates_rows = instrument_risk[instrument_risk["InstrumentType"].isin(["Zero Coupon Bond", "Fixed Rate Bond", "Interest Rate Swap"])]
    assert not rates_rows.empty
    assert rates_rows["NPV"].notna().all()
    assert (rates_rows["NPV"].abs() > 0).all()

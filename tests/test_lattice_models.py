from __future__ import annotations

import pandas as pd

from src.pricing.fallback_bs import black_scholes_price
from src.pricing.lattice_models import (
    lattice_convergence_analysis,
    price_european_option_binomial,
    price_european_option_trinomial,
)


def test_binomial_converges_to_black_scholes_call() -> None:
    bs = black_scholes_price(
        spot=100.0,
        strike=100.0,
        maturity_years=1.0,
        rate=0.05,
        volatility=0.2,
        option_type="call",
    )
    tree = price_european_option_binomial(
        spot=100.0,
        strike=100.0,
        maturity_years=1.0,
        rate=0.05,
        volatility=0.2,
        steps=500,
        option_type="call",
    )
    assert 0.0 <= tree.risk_neutral_probability <= 1.0
    assert abs(bs - tree.price) < 0.10


def test_trinomial_converges_to_black_scholes_put() -> None:
    bs = black_scholes_price(
        spot=100.0,
        strike=100.0,
        maturity_years=1.0,
        rate=0.05,
        volatility=0.2,
        option_type="put",
    )
    tri = price_european_option_trinomial(
        spot=100.0,
        strike=100.0,
        maturity_years=1.0,
        rate=0.05,
        volatility=0.2,
        steps=500,
        option_type="put",
    )
    assert abs(tri.up_probability + tri.middle_probability + tri.down_probability - 1.0) < 1e-12
    assert abs(bs - tri.price) < 0.10


def test_lattice_convergence_analysis_returns_binomial_and_trinomial_rows() -> None:
    report = lattice_convergence_analysis(
        spot=100.0,
        strike=100.0,
        maturity_years=1.0,
        rate=0.05,
        volatility=0.2,
        option_type="Call",
        step_counts=(25, 50, 100),
    )
    assert isinstance(report, pd.DataFrame)
    assert set(report["model"]) == {"binomial", "trinomial"}
    assert set(report["steps"]) == {25, 50, 100}
    assert (report["absolute_error"] >= 0.0).all()
    assert report.loc[report["steps"].eq(100), "absolute_error"].max() < 0.25

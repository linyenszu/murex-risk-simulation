from __future__ import annotations

import numpy as np

from src.pricing.fallback_bs import black_scholes_greeks
from src.pricing.monte_carlo import (
    convergence_analysis,
    european_option_payoff,
    price_european_option_mc,
    simulate_gbm_paths,
    simulate_gbm_terminal_prices,
)


def test_european_option_payoff_call_and_put() -> None:
    terminal = np.array([80.0, 100.0, 120.0])
    assert np.allclose(european_option_payoff(terminal, 100.0, "Call"), [0.0, 0.0, 20.0])
    assert np.allclose(european_option_payoff(terminal, 100.0, "Put"), [20.0, 0.0, 0.0])


def test_gbm_terminal_prices_are_positive_and_reproducible() -> None:
    p1 = simulate_gbm_terminal_prices(100.0, 1.0, 0.05, 0.2, 10_000, seed=123)
    p2 = simulate_gbm_terminal_prices(100.0, 1.0, 0.05, 0.2, 10_000, seed=123)
    assert p1.shape == (10_000,)
    assert np.all(p1 > 0.0)
    assert np.allclose(p1, p2)


def test_gbm_paths_shape_and_initial_spot() -> None:
    paths = simulate_gbm_paths(
        spot=100.0,
        maturity_years=1.0,
        rate=0.05,
        volatility=0.2,
        num_paths=250,
        num_steps=12,
        seed=7,
    )
    assert paths.shape == (250, 13)
    assert np.allclose(paths[:, 0], 100.0)
    assert np.all(paths > 0.0)


def test_monte_carlo_call_price_converges_near_black_scholes() -> None:
    mc = price_european_option_mc(
        spot=100.0,
        strike=100.0,
        maturity_years=1.0,
        rate=0.05,
        volatility=0.2,
        option_type="Call",
        num_paths=200_000,
        seed=42,
        antithetic=True,
    )
    bs = black_scholes_greeks(100.0, 100.0, 1.0, 0.05, 0.2, "Call")
    assert abs(mc.price - bs.price) < 0.15
    assert mc.confidence_interval_low < bs.price < mc.confidence_interval_high


def test_monte_carlo_put_price_converges_near_black_scholes() -> None:
    mc = price_european_option_mc(
        spot=100.0,
        strike=105.0,
        maturity_years=0.75,
        rate=0.04,
        volatility=0.25,
        option_type="Put",
        num_paths=200_000,
        seed=99,
        antithetic=True,
    )
    bs = black_scholes_greeks(100.0, 105.0, 0.75, 0.04, 0.25, "Put")
    assert abs(mc.price - bs.price) < 0.20


def test_convergence_analysis_schema_and_error_reduction_signal() -> None:
    df = convergence_analysis(
        spot=100.0,
        strike=100.0,
        maturity_years=1.0,
        rate=0.05,
        volatility=0.2,
        option_type="Call",
        path_counts=(1_000, 5_000, 20_000),
        seed=123,
    )
    expected = {
        "num_paths",
        "mc_price",
        "standard_error",
        "ci_low",
        "ci_high",
        "black_scholes_price",
        "absolute_error",
        "relative_error_pct",
    }
    assert expected.issubset(df.columns)
    assert list(df["num_paths"]) == [1_000, 5_000, 20_000]
    assert df["standard_error"].iloc[-1] < df["standard_error"].iloc[0]

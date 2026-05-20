"""Monte Carlo pricing engine for European options.

The implementation is intentionally dependency-light and deterministic under a
seed so it can be used in unit tests, notebooks, CI, and interview/demo repos.
It supports risk-neutral GBM terminal simulation, optional full path generation,
antithetic variates, Black-Scholes comparison, and convergence diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from typing import Literal

import numpy as np
import pandas as pd

from src.pricing.fallback_bs import black_scholes_greeks

OptionType = Literal["Call", "Put", "call", "put"]


@dataclass(frozen=True)
class MonteCarloOptionResult:
    """Result container for Monte Carlo European option pricing."""

    price: float
    standard_error: float
    confidence_interval_low: float
    confidence_interval_high: float
    num_paths: int
    maturity_years: float
    spot: float
    strike: float
    rate: float
    volatility: float
    dividend_yield: float
    option_type: str
    seed: int | None
    antithetic: bool

    @property
    def ci_width(self) -> float:
        """Return the two-sided 95% confidence interval width."""
        return self.confidence_interval_high - self.confidence_interval_low


def _validate_inputs(
    spot: float,
    strike: float,
    maturity_years: float,
    rate: float,
    volatility: float,
    num_paths: int,
    option_type: str,
) -> None:
    if spot <= 0:
        raise ValueError("spot must be positive")
    if strike <= 0:
        raise ValueError("strike must be positive")
    if maturity_years <= 0:
        raise ValueError("maturity_years must be positive")
    if volatility <= 0:
        raise ValueError("volatility must be positive")
    if num_paths < 2:
        raise ValueError("num_paths must be at least 2")
    if option_type.lower() not in {"call", "put"}:
        raise ValueError("option_type must be 'Call' or 'Put'")
    # Negative rates are allowed in production markets, so no validation here.
    _ = rate


def generate_standard_normals(
    num_paths: int,
    rng: np.random.Generator,
    antithetic: bool = True,
) -> np.ndarray:
    """Generate standard normal shocks, optionally with antithetic variates.

    Antithetic sampling pairs every ``z`` with ``-z``. This usually reduces the
    variance of option-price estimates while preserving the desired number of
    terminal draws.
    """
    if antithetic:
        half = (num_paths + 1) // 2
        z_half = rng.standard_normal(half)
        z = np.concatenate([z_half, -z_half])[:num_paths]
        return z
    return rng.standard_normal(num_paths)


def simulate_gbm_terminal_prices(
    spot: float,
    maturity_years: float,
    rate: float,
    volatility: float,
    num_paths: int,
    dividend_yield: float = 0.0,
    seed: int | None = 42,
    antithetic: bool = True,
) -> np.ndarray:
    """Simulate terminal prices under risk-neutral geometric Brownian motion.

    Under the risk-neutral measure:

    ``S_T = S_0 * exp((r - q - 0.5*sigma^2)T + sigma*sqrt(T)*Z)``

    where ``q`` is a continuous dividend yield and ``Z`` is standard normal.
    """
    _validate_inputs(spot, spot, maturity_years, rate, volatility, num_paths, "Call")
    rng = np.random.default_rng(seed)
    z = generate_standard_normals(num_paths, rng, antithetic=antithetic)
    drift = (rate - dividend_yield - 0.5 * volatility**2) * maturity_years
    diffusion = volatility * sqrt(maturity_years) * z
    return spot * np.exp(drift + diffusion)


def simulate_gbm_paths(
    spot: float,
    maturity_years: float,
    rate: float,
    volatility: float,
    num_paths: int,
    num_steps: int,
    dividend_yield: float = 0.0,
    seed: int | None = 42,
) -> np.ndarray:
    """Generate full GBM price paths for diagnostics and visualization.

    Returns an array of shape ``(num_paths, num_steps + 1)``. The first column is
    the initial spot. Full paths are not required for European option pricing,
    but they are useful for explaining random path generation and for extending
    the engine to path-dependent products.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    _validate_inputs(spot, spot, maturity_years, rate, volatility, num_paths, "Call")
    rng = np.random.default_rng(seed)
    dt = maturity_years / num_steps
    shocks = rng.standard_normal((num_paths, num_steps))
    increments = (rate - dividend_yield - 0.5 * volatility**2) * dt + volatility * sqrt(dt) * shocks
    log_paths = np.cumsum(increments, axis=1)
    paths = np.empty((num_paths, num_steps + 1), dtype=float)
    paths[:, 0] = spot
    paths[:, 1:] = spot * np.exp(log_paths)
    return paths


def european_option_payoff(terminal_prices: np.ndarray, strike: float, option_type: OptionType) -> np.ndarray:
    """Return European option payoff at maturity."""
    opt = option_type.lower()
    if opt == "call":
        return np.maximum(terminal_prices - strike, 0.0)
    if opt == "put":
        return np.maximum(strike - terminal_prices, 0.0)
    raise ValueError("option_type must be 'Call' or 'Put'")


def price_european_option_mc(
    spot: float,
    strike: float,
    maturity_years: float,
    rate: float,
    volatility: float,
    option_type: OptionType = "Call",
    num_paths: int = 100_000,
    dividend_yield: float = 0.0,
    seed: int | None = 42,
    antithetic: bool = True,
    confidence_z: float = 1.96,
) -> MonteCarloOptionResult:
    """Price a European option by Monte Carlo risk-neutral valuation."""
    _validate_inputs(spot, strike, maturity_years, rate, volatility, num_paths, option_type)
    terminal_prices = simulate_gbm_terminal_prices(
        spot=spot,
        maturity_years=maturity_years,
        rate=rate,
        volatility=volatility,
        num_paths=num_paths,
        dividend_yield=dividend_yield,
        seed=seed,
        antithetic=antithetic,
    )
    payoff = european_option_payoff(terminal_prices, strike, option_type)
    discounted_payoff = exp(-rate * maturity_years) * payoff
    price = float(np.mean(discounted_payoff))
    standard_error = float(np.std(discounted_payoff, ddof=1) / sqrt(num_paths))
    ci_half_width = confidence_z * standard_error
    return MonteCarloOptionResult(
        price=price,
        standard_error=standard_error,
        confidence_interval_low=price - ci_half_width,
        confidence_interval_high=price + ci_half_width,
        num_paths=num_paths,
        maturity_years=maturity_years,
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        option_type=option_type.capitalize(),
        seed=seed,
        antithetic=antithetic,
    )


def convergence_analysis(
    spot: float,
    strike: float,
    maturity_years: float,
    rate: float,
    volatility: float,
    option_type: OptionType = "Call",
    path_counts: list[int] | tuple[int, ...] = (1_000, 5_000, 10_000, 25_000, 50_000, 100_000),
    dividend_yield: float = 0.0,
    seed: int | None = 42,
    antithetic: bool = True,
) -> pd.DataFrame:
    """Compare Monte Carlo estimates across path counts against Black-Scholes."""
    bs = black_scholes_greeks(
        spot=spot,
        strike=strike,
        maturity_years=maturity_years,
        rate=rate,
        volatility=volatility,
        option_type=option_type,
        dividend_yield=dividend_yield,
    )
    rows: list[dict[str, float | int | str | bool | None]] = []
    for n_paths in path_counts:
        result = price_european_option_mc(
            spot=spot,
            strike=strike,
            maturity_years=maturity_years,
            rate=rate,
            volatility=volatility,
            option_type=option_type,
            num_paths=int(n_paths),
            dividend_yield=dividend_yield,
            seed=seed,
            antithetic=antithetic,
        )
        rows.append(
            {
                "num_paths": result.num_paths,
                "mc_price": result.price,
                "standard_error": result.standard_error,
                "ci_low": result.confidence_interval_low,
                "ci_high": result.confidence_interval_high,
                "black_scholes_price": bs.price,
                "absolute_error": abs(result.price - bs.price),
                "relative_error_pct": abs(result.price - bs.price) / bs.price * 100.0 if bs.price else 0.0,
                "option_type": result.option_type,
                "antithetic": result.antithetic,
                "seed": result.seed,
            }
        )
    return pd.DataFrame(rows)

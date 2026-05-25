"""Lattice/tree pricing models for European options under risk-neutral GBM.

The binomial and trinomial engines approximate the same risk-neutral geometric
Brownian motion dynamics used by Black-Scholes:

    dS_t = (r - q) S_t dt + sigma S_t dW_t

They are useful for explaining risk-neutral probabilities, backward induction,
and convergence to the analytic Black-Scholes benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from typing import Iterable, Literal

import pandas as pd

from src.pricing.fallback_bs import black_scholes_greeks

OptionType = Literal["Call", "Put", "call", "put"]


@dataclass(frozen=True, slots=True)
class BinomialTreeResult:
    price: float
    up_factor: float
    down_factor: float
    risk_neutral_probability: float
    steps: int
    dt: float


@dataclass(frozen=True, slots=True)
class TrinomialTreeResult:
    price: float
    up_factor: float
    down_factor: float
    middle_factor: float
    up_probability: float
    middle_probability: float
    down_probability: float
    steps: int
    dt: float


def _validate_tree_inputs(
    spot: float,
    strike: float,
    maturity_years: float,
    rate: float,
    volatility: float,
    steps: int,
    option_type: str,
) -> None:
    if spot <= 0.0:
        raise ValueError("spot must be positive")
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if maturity_years <= 0.0:
        raise ValueError("maturity_years must be positive")
    if volatility <= 0.0:
        raise ValueError("volatility must be positive")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if option_type.lower() not in {"call", "put"}:
        raise ValueError("option_type must be 'Call' or 'Put'")
    _ = rate


def _payoff(spot: float, strike: float, option_type: str) -> float:
    if option_type.lower() == "call":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def price_european_option_binomial(
    spot: float,
    strike: float,
    maturity_years: float | None = None,
    rate: float = 0.0,
    volatility: float = 0.2,
    steps: int = 200,
    option_type: OptionType = "Call",
    dividend_yield: float = 0.0,
    maturity: float | None = None,
) -> BinomialTreeResult:
    """Price a European option with a Cox-Ross-Rubinstein binomial tree.

    The CRR tree is a discrete risk-neutral GBM approximation. Continuous
    dividend yield ``q`` is supported through the risk-neutral drift ``r - q``.
    ``maturity`` is accepted as a backward-compatible alias for
    ``maturity_years``.
    """
    if maturity_years is None:
        if maturity is None:
            raise ValueError("maturity_years is required")
        maturity_years = maturity
    _validate_tree_inputs(spot, strike, maturity_years, rate, volatility, steps, option_type)

    dt = maturity_years / steps
    up = exp(volatility * sqrt(dt))
    down = 1.0 / up
    growth = exp((rate - dividend_yield) * dt)
    discount = exp(-rate * dt)
    probability = (growth - down) / (up - down)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "Invalid binomial risk-neutral probability. Increase steps or check rate/dividend/volatility inputs."
        )

    values = [
        _payoff(spot * (up ** (steps - i)) * (down ** i), strike, option_type)
        for i in range(steps + 1)
    ]
    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            values[i] = discount * (probability * values[i] + (1.0 - probability) * values[i + 1])

    return BinomialTreeResult(
        price=float(values[0]),
        up_factor=up,
        down_factor=down,
        risk_neutral_probability=probability,
        steps=steps,
        dt=dt,
    )


def price_european_option_trinomial(
    spot: float,
    strike: float,
    maturity_years: float | None = None,
    rate: float = 0.0,
    volatility: float = 0.2,
    steps: int = 100,
    option_type: OptionType = "Call",
    dividend_yield: float = 0.0,
    maturity: float | None = None,
) -> TrinomialTreeResult:
    """Price a European option with a recombining trinomial GBM tree.

    Uses a moment-matched trinomial scheme for log prices under the
    risk-neutral drift ``r - q``. Returns probabilities and tree factors for
    model diagnostics.
    """
    if maturity_years is None:
        if maturity is None:
            raise ValueError("maturity_years is required")
        maturity_years = maturity
    _validate_tree_inputs(spot, strike, maturity_years, rate, volatility, steps, option_type)

    dt = maturity_years / steps
    drift = rate - dividend_yield - 0.5 * volatility**2
    dx = volatility * sqrt(3.0 * dt)
    variance_term = volatility**2 * dt + drift**2 * dt**2
    up_probability = 0.5 * (variance_term / dx**2 + drift * dt / dx)
    down_probability = 0.5 * (variance_term / dx**2 - drift * dt / dx)
    middle_probability = 1.0 - up_probability - down_probability
    if min(up_probability, middle_probability, down_probability) < -1e-12:
        raise ValueError(
            "Invalid trinomial risk-neutral probabilities. Increase steps or check rate/dividend/volatility inputs."
        )
    up_probability = max(0.0, up_probability)
    middle_probability = max(0.0, middle_probability)
    down_probability = max(0.0, down_probability)

    up = exp(dx)
    middle = 1.0
    down = exp(-dx)
    discount = exp(-rate * dt)

    values: dict[int, float] = {
        j: _payoff(spot * exp(j * dx), strike, option_type) for j in range(-steps, steps + 1)
    }
    for step in range(steps - 1, -1, -1):
        new_values: dict[int, float] = {}
        for j in range(-step, step + 1):
            new_values[j] = discount * (
                up_probability * values[j + 1]
                + middle_probability * values[j]
                + down_probability * values[j - 1]
            )
        values = new_values

    return TrinomialTreeResult(
        price=float(values[0]),
        up_factor=up,
        down_factor=down,
        middle_factor=middle,
        up_probability=up_probability,
        middle_probability=middle_probability,
        down_probability=down_probability,
        steps=steps,
        dt=dt,
    )


def lattice_convergence_analysis(
    spot: float,
    strike: float,
    maturity_years: float,
    rate: float,
    volatility: float,
    option_type: OptionType = "Call",
    step_counts: Iterable[int] = (5, 10, 25, 50, 100, 200, 500),
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """Compare binomial/trinomial prices against Black-Scholes by tree size."""
    bs = black_scholes_greeks(
        spot=spot,
        strike=strike,
        maturity_years=maturity_years,
        rate=rate,
        volatility=volatility,
        option_type=option_type,
        dividend_yield=dividend_yield,
    )
    rows: list[dict[str, float | int | str]] = []
    for steps in step_counts:
        n = int(steps)
        binomial = price_european_option_binomial(
            spot=spot,
            strike=strike,
            maturity_years=maturity_years,
            rate=rate,
            volatility=volatility,
            steps=n,
            option_type=option_type,
            dividend_yield=dividend_yield,
        )
        trinomial = price_european_option_trinomial(
            spot=spot,
            strike=strike,
            maturity_years=maturity_years,
            rate=rate,
            volatility=volatility,
            steps=n,
            option_type=option_type,
            dividend_yield=dividend_yield,
        )
        for model_name, price in (
            ("binomial", binomial.price),
            ("trinomial", trinomial.price),
        ):
            rows.append(
                {
                    "model": model_name,
                    "steps": n,
                    "tree_price": price,
                    "black_scholes_price": bs.price,
                    "absolute_error": abs(price - bs.price),
                    "relative_error_pct": abs(price - bs.price) / bs.price * 100.0 if bs.price else 0.0,
                    "option_type": option_type.capitalize(),
                }
            )
    return pd.DataFrame(rows)

"""Pure-Python Black-Scholes pricing and Greeks fallback.

Used when QuantLib-Python is not installed. Values are intended for demo and
unit-test stability, not model validation sign-off.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


@dataclass(frozen=True)
class BlackScholesResult:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def black_scholes_greeks(
    spot: float,
    strike: float,
    maturity_years: float,
    rate: float,
    volatility: float,
    option_type: str,
    dividend_yield: float = 0.0,
) -> BlackScholesResult:
    """Return Black-Scholes price and Greeks for a European call/put."""
    if spot <= 0 or strike <= 0 or volatility <= 0 or maturity_years <= 0:
        raise ValueError("spot, strike, volatility, and maturity must be positive")

    t = maturity_years
    sigma_sqrt_t = volatility * sqrt(t)
    d1 = (log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * t) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    df_r = exp(-rate * t)
    df_q = exp(-dividend_yield * t)
    opt = option_type.lower()

    if opt == "call":
        price = spot * df_q * _cdf(d1) - strike * df_r * _cdf(d2)
        delta = df_q * _cdf(d1)
        theta = (-(spot * df_q * _pdf(d1) * volatility) / (2 * sqrt(t))
                 - rate * strike * df_r * _cdf(d2)
                 + dividend_yield * spot * df_q * _cdf(d1)) / 365.0
        rho = strike * t * df_r * _cdf(d2)
    elif opt == "put":
        price = strike * df_r * _cdf(-d2) - spot * df_q * _cdf(-d1)
        delta = df_q * (_cdf(d1) - 1.0)
        theta = (-(spot * df_q * _pdf(d1) * volatility) / (2 * sqrt(t))
                 + rate * strike * df_r * _cdf(-d2)
                 - dividend_yield * spot * df_q * _cdf(-d1)) / 365.0
        rho = -strike * t * df_r * _cdf(-d2)
    else:
        raise ValueError("option_type must be 'Call' or 'Put'")

    gamma = df_q * _pdf(d1) / (spot * sigma_sqrt_t)
    vega = spot * df_q * _pdf(d1) * sqrt(t) / 100.0
    rho = rho / 100.0
    return BlackScholesResult(price, delta, gamma, vega, theta, rho)


def black_scholes_price(
    spot: float,
    strike: float,
    maturity_years: float | None = None,
    rate: float = 0.0,
    volatility: float = 0.2,
    option_type: str = "Call",
    dividend_yield: float = 0.0,
    maturity: float | None = None,
) -> float:
    """Backward-compatible convenience wrapper returning only BS price."""
    if maturity_years is None:
        if maturity is None:
            raise ValueError("maturity_years is required")
        maturity_years = maturity
    return black_scholes_greeks(
        spot=spot,
        strike=strike,
        maturity_years=maturity_years,
        rate=rate,
        volatility=volatility,
        option_type=option_type,
        dividend_yield=dividend_yield,
    ).price

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt
from typing import Any
import pandas as pd


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


@dataclass(frozen=True)
class MarketContext:
    valuation_date: pd.Timestamp
    risk_free_rate: float
    dividend_yield: float
    vols: dict[str, float]
    fx_foreign_rates: dict[str, float]
    zero_curve: Any | None = None

    def time_to_maturity(self, maturity: pd.Timestamp) -> float:
        return max((pd.Timestamp(maturity) - self.valuation_date).days / 365.0, 0.0)

    def discount_factor(self, maturity_years: float) -> float:
        """Return discount factor from zero curve when available, else flat rate."""
        t = max(float(maturity_years), 0.0)
        if t == 0.0:
            return 1.0
        if self.zero_curve is not None:
            return float(self.zero_curve.discount_factor(t))
        return exp(-self.risk_free_rate * t)

    def zero_rate(self, maturity_years: float) -> float:
        """Return continuously compounded zero rate for maturity."""
        t = max(float(maturity_years), 0.0)
        if t == 0.0:
            return self.risk_free_rate
        if self.zero_curve is not None:
            return float(self.zero_curve.zero_rate(t))
        return self.risk_free_rate

    def forward_rate(self, start_years: float, end_years: float) -> float:
        """Return continuously compounded forward rate from the curve."""
        start = float(start_years)
        end = float(end_years)
        if not 0 <= start < end:
            raise ValueError("Require 0 <= start_years < end_years")
        if self.zero_curve is not None:
            return float(self.zero_curve.forward_rate(start, end))
        return self.risk_free_rate


def black_scholes_price_greeks(
    spot: float,
    strike: float,
    maturity_years: float,
    rate: float,
    dividend: float,
    vol: float,
    option_type: str,
) -> dict[str, float]:
    if maturity_years <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        intrinsic = max(spot - strike, 0.0) if option_type == "Call" else max(strike - spot, 0.0)
        return {"NPV": intrinsic, "Delta": 0.0, "Gamma": 0.0, "Vega": 0.0, "Theta": 0.0, "Rho": 0.0}

    d1 = (log(spot / strike) + (rate - dividend + 0.5 * vol * vol) * maturity_years) / (vol * sqrt(maturity_years))
    d2 = d1 - vol * sqrt(maturity_years)
    disc_r = exp(-rate * maturity_years)
    disc_q = exp(-dividend * maturity_years)

    if option_type == "Call":
        price = spot * disc_q * _norm_cdf(d1) - strike * disc_r * _norm_cdf(d2)
        delta = disc_q * _norm_cdf(d1)
        theta = (-(spot * disc_q * _norm_pdf(d1) * vol) / (2 * sqrt(maturity_years)) - rate * strike * disc_r * _norm_cdf(d2) + dividend * spot * disc_q * _norm_cdf(d1)) / 365.0
        rho = strike * maturity_years * disc_r * _norm_cdf(d2)
    else:
        price = strike * disc_r * _norm_cdf(-d2) - spot * disc_q * _norm_cdf(-d1)
        delta = -disc_q * _norm_cdf(-d1)
        theta = (-(spot * disc_q * _norm_pdf(d1) * vol) / (2 * sqrt(maturity_years)) + rate * strike * disc_r * _norm_cdf(-d2) - dividend * spot * disc_q * _norm_cdf(-d1)) / 365.0
        rho = -strike * maturity_years * disc_r * _norm_cdf(-d2)

    gamma = disc_q * _norm_pdf(d1) / (spot * vol * sqrt(maturity_years))
    vega = spot * disc_q * _norm_pdf(d1) * sqrt(maturity_years) / 100.0
    return {"NPV": price, "Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho / 100.0}


def price_stock(quantity: float, spot: float) -> dict[str, float]:
    return {"NPV": quantity * spot, "Delta": quantity, "Gamma": 0.0, "Vega": 0.0, "Theta": 0.0, "Rho": 0.0}


def price_fx_forward(quantity: float, spot: float, strike: float, maturity: pd.Timestamp, ticker: str, ctx: MarketContext) -> dict[str, float]:
    t = ctx.time_to_maturity(maturity)
    foreign_rate = ctx.fx_foreign_rates.get(ticker, 0.0)
    domestic_df = ctx.discount_factor(t)
    foreign_df = exp(-foreign_rate * t)
    fwd = spot * foreign_df / domestic_df
    npv = quantity * (fwd - strike) * domestic_df
    delta = quantity * foreign_df
    return {"NPV": npv, "Delta": delta, "Gamma": 0.0, "Vega": 0.0, "Theta": 0.0, "Rho": 0.0}


def price_equity_future(quantity: float, spot: float, strike: float, maturity: pd.Timestamp, ctx: MarketContext, multiplier: float = 1.0) -> dict[str, float]:
    """Mark an equity future as discounted forward payoff.

    Positive quantity means long futures exposure. Strike is the contract price.
    """
    t = ctx.time_to_maturity(maturity)
    discount = ctx.discount_factor(t)
    npv = quantity * multiplier * (spot - strike) * discount
    delta = quantity * multiplier * discount
    theta = -ctx.risk_free_rate * npv / 365.0
    return {"NPV": npv, "Delta": delta, "Gamma": 0.0, "Vega": 0.0, "Theta": theta, "Rho": -t * npv / 100.0}


def price_zero_coupon_bond(quantity: float, face_value: float, maturity: pd.Timestamp, ctx: MarketContext, yield_rate: float | None = None) -> dict[str, float]:
    t = ctx.time_to_maturity(maturity)
    y = ctx.zero_rate(t) if yield_rate is None else yield_rate
    unit_pv = face_value * (ctx.discount_factor(t) if yield_rate is None else exp(-y * t))
    npv = quantity * unit_pv
    # Delta here is dollar sensitivity to one absolute rate point. Rho is per 1 bp.
    rate_delta = -t * npv
    return {"NPV": npv, "Delta": rate_delta, "Gamma": t * t * npv, "Vega": 0.0, "Theta": y * npv / 365.0, "Rho": rate_delta / 100.0}


def price_fixed_rate_bond(
    quantity: float,
    face_value: float,
    coupon_rate: float,
    maturity: pd.Timestamp,
    ctx: MarketContext,
    yield_rate: float | None = None,
    frequency: int = 2,
) -> dict[str, float]:
    t = ctx.time_to_maturity(maturity)
    y = ctx.zero_rate(t) if yield_rate is None else yield_rate
    if t <= 0:
        return {"NPV": 0.0, "Delta": 0.0, "Gamma": 0.0, "Vega": 0.0, "Theta": 0.0, "Rho": 0.0}
    n_periods = max(int(round(t * frequency)), 1)
    period_rate = y / frequency
    coupon = face_value * coupon_rate / frequency
    pv = 0.0
    duration_weight = 0.0
    convexity_weight = 0.0
    for i in range(1, n_periods + 1):
        cash_flow = coupon + (face_value if i == n_periods else 0.0)
        time_i = i / frequency
        df = ctx.discount_factor(time_i) if yield_rate is None else (1.0 + period_rate) ** (-i)
        pv += cash_flow * df
        duration_weight += time_i * cash_flow * df
        convexity_weight += time_i * time_i * cash_flow * df
    npv = quantity * pv
    rate_delta = -quantity * duration_weight
    gamma = quantity * convexity_weight
    return {"NPV": npv, "Delta": rate_delta, "Gamma": gamma, "Vega": 0.0, "Theta": y * npv / 365.0, "Rho": rate_delta / 100.0}


def price_interest_rate_swap(
    quantity: float,
    notional: float,
    fixed_rate: float,
    maturity: pd.Timestamp,
    ctx: MarketContext,
    floating_rate: float,
    pay_receive: str = "Payer",
    frequency: int = 2,
) -> dict[str, float]:
    """Approximate vanilla IRS PV using par-rate spread times fixed-leg annuity.

    Payer means pay fixed / receive floating, so PV increases when floating/par rate rises.
    Receiver has the opposite sign.
    """
    t = ctx.time_to_maturity(maturity)
    if t <= 0:
        return {"NPV": 0.0, "Delta": 0.0, "Gamma": 0.0, "Vega": 0.0, "Theta": 0.0, "Rho": 0.0}
    n_periods = max(int(round(t * frequency)), 1)
    annuity = sum((1.0 / frequency) * ctx.discount_factor(i / frequency) for i in range(1, n_periods + 1))
    direction = 1.0 if str(pay_receive).lower().startswith("payer") else -1.0
    par_rate = ctx.forward_rate(0.0, t)
    effective_float_rate = par_rate if floating_rate is None else floating_rate
    npv = quantity * direction * notional * (effective_float_rate - fixed_rate) * annuity
    delta = quantity * direction * notional * annuity
    return {"NPV": npv, "Delta": delta, "Gamma": 0.0, "Vega": 0.0, "Theta": -ctx.risk_free_rate * npv / 365.0, "Rho": delta / 100.0}

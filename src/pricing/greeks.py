from __future__ import annotations

import numpy as np
import pandas as pd
from src.pricing.instruments import (
    MarketContext,
    black_scholes_price_greeks,
    price_equity_future,
    price_fixed_rate_bond,
    price_fx_forward,
    price_interest_rate_swap,
    price_stock,
    price_zero_coupon_bond,
)

GREEK_COLUMNS = ["NPV", "Delta", "Gamma", "Vega", "Theta", "Rho"]


def _optional_float(row: pd.Series, name: str, default: float) -> float:
    value = row.get(name, default)
    if pd.isna(value):
        return default
    return float(value)


def _optional_str(row: pd.Series, name: str, default: str) -> str:
    value = row.get(name, default)
    if pd.isna(value):
        return default
    return str(value)


def price_position(row: pd.Series, ctx: MarketContext, override_spot: float | None = None) -> dict[str, float]:
    spot = float(row["CurrentPrice"] if override_spot is None else override_spot)
    quantity = float(row["Quantity"])
    instrument_type = row["InstrumentType"]

    if instrument_type == "Stock":
        return price_stock(quantity, spot)

    if instrument_type == "FX Forward":
        return price_fx_forward(quantity, spot, float(row["Strike"]), row["Maturity"], row["Ticker"], ctx)

    if instrument_type == "European Option":
        unit = black_scholes_price_greeks(
            spot=spot,
            strike=float(row["Strike"]),
            maturity_years=ctx.time_to_maturity(row["Maturity"]),
            rate=ctx.risk_free_rate,
            dividend=ctx.dividend_yield,
            vol=ctx.vols.get(row["Ticker"], 0.20),
            option_type=str(row["OptionType"]),
        )
        return {k: v * quantity for k, v in unit.items()}

    if instrument_type == "Equity Future":
        return price_equity_future(
            quantity=quantity,
            spot=spot,
            strike=float(row["Strike"]),
            maturity=row["Maturity"],
            ctx=ctx,
            multiplier=_optional_float(row, "ContractMultiplier", 1.0),
        )

    if instrument_type == "Zero Coupon Bond":
        return price_zero_coupon_bond(
            quantity=quantity,
            face_value=_optional_float(row, "FaceValue", 1000.0),
            maturity=row["Maturity"],
            ctx=ctx,
            yield_rate=spot,
        )

    if instrument_type == "Fixed Rate Bond":
        return price_fixed_rate_bond(
            quantity=quantity,
            face_value=_optional_float(row, "FaceValue", 1000.0),
            coupon_rate=_optional_float(row, "CouponRate", 0.04),
            maturity=row["Maturity"],
            ctx=ctx,
            yield_rate=spot,
            frequency=int(_optional_float(row, "PaymentFrequency", 2.0)),
        )

    if instrument_type == "Interest Rate Swap":
        return price_interest_rate_swap(
            quantity=quantity,
            notional=_optional_float(row, "Notional", 1_000_000.0),
            fixed_rate=_optional_float(row, "FixedRate", 0.03),
            maturity=row["Maturity"],
            ctx=ctx,
            floating_rate=spot,
            pay_receive=_optional_str(row, "PayReceive", "Payer"),
            frequency=int(_optional_float(row, "PaymentFrequency", 2.0)),
        )

    return {k: np.nan for k in GREEK_COLUMNS}


def calculate_instrument_risk(positions: pd.DataFrame, ctx: MarketContext) -> pd.DataFrame:
    priced = positions.copy()
    rows = [price_position(row, ctx) for _, row in priced.iterrows()]
    risk = pd.DataFrame(rows, index=priced.index)
    for col in GREEK_COLUMNS:
        priced[col] = risk[col]
    return priced

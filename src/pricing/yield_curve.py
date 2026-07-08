from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ZeroCurvePoint:
    """A bootstrapped zero-coupon curve pillar.

    Attributes:
        maturity_years: Time to maturity in years.
        discount_factor: Continuously compounded discount factor.
        zero_rate: Continuously compounded annual zero rate.
    """

    maturity_years: float
    discount_factor: float
    zero_rate: float


REQUIRED_BOND_COLUMNS = {
    "InstrumentID",
    "MaturityYears",
    "CouponRate",
    "MarketPrice",
}


def sample_bond_market_quotes() -> pd.DataFrame:
    """Return deterministic example Treasury-style bond quotes.

    Prices are quoted per 100 notional and assume valuation occurs on a coupon
    date, so clean price equals dirty price. This keeps the example focused on
    bootstrapping mechanics rather than accrued-interest conventions.
    """

    return pd.DataFrame(
        {
            "InstrumentID": ["BOND_6M", "BOND_1Y", "BOND_2Y", "BOND_3Y", "BOND_5Y", "BOND_7Y", "BOND_10Y"],
            "MaturityYears": [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
            "CouponRate": [0.0000, 0.0000, 0.0350, 0.0375, 0.0400, 0.0425, 0.0450],
            "MarketPrice": [98.90, 97.55, 99.10, 99.25, 100.20, 100.35, 101.15],
            "FaceValue": [100.0] * 7,
            "Frequency": [2] * 7,
        }
    )


def validate_bond_quotes(bonds: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize bond quote input.

    Expected columns:
        InstrumentID, MaturityYears, CouponRate, MarketPrice

    Optional columns:
        FaceValue, Frequency
    """

    missing = REQUIRED_BOND_COLUMNS.difference(bonds.columns)
    if missing:
        raise ValueError(f"Missing required bond quote columns: {sorted(missing)}")

    quotes = bonds.copy()
    if "FaceValue" not in quotes.columns:
        quotes["FaceValue"] = 100.0
    if "Frequency" not in quotes.columns:
        quotes["Frequency"] = 2

    numeric_cols = ["MaturityYears", "CouponRate", "MarketPrice", "FaceValue", "Frequency"]
    for col in numeric_cols:
        quotes[col] = pd.to_numeric(quotes[col], errors="raise")

    if (quotes["MaturityYears"] <= 0).any():
        raise ValueError("MaturityYears must be strictly positive")
    if (quotes["MarketPrice"] <= 0).any():
        raise ValueError("MarketPrice must be strictly positive")
    if (quotes["FaceValue"] <= 0).any():
        raise ValueError("FaceValue must be strictly positive")
    if (quotes["Frequency"] <= 0).any():
        raise ValueError("Frequency must be strictly positive")
    if quotes["MaturityYears"].duplicated().any():
        duplicates = quotes.loc[quotes["MaturityYears"].duplicated(), "MaturityYears"].tolist()
        raise ValueError(f"Duplicate maturity pillars are not supported: {duplicates}")

    return quotes.sort_values("MaturityYears").reset_index(drop=True)


def _cashflow_times(maturity_years: float, frequency: int) -> list[float]:
    n_periods = max(int(round(maturity_years * frequency)), 1)
    return [i / frequency for i in range(1, n_periods + 1)]


def _interpolated_discount_factor(time_years: float, bootstrapped: dict[float, float]) -> float:
    """Interpolate discount factors using log-linear interpolation.

    Log-linear discount-factor interpolation is common for continuously
    compounded curves because it corresponds to linear interpolation of zero
    rates over short intervals.
    """

    if time_years in bootstrapped:
        return bootstrapped[time_years]
    if not bootstrapped:
        raise ValueError("Cannot interpolate discount factor before first bootstrapped pillar")

    pillars = np.array(sorted(bootstrapped), dtype=float)
    log_dfs = np.array([log(bootstrapped[t]) for t in pillars], dtype=float)

    if time_years < pillars[0]:
        zero_rate = -log_dfs[0] / pillars[0]
        return exp(-zero_rate * time_years)
    if time_years > pillars[-1]:
        zero_rate = -log_dfs[-1] / pillars[-1]
        return exp(-zero_rate * time_years)

    return float(exp(np.interp(time_years, pillars, log_dfs)))


def bootstrap_zero_coupon_curve(bonds: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap a zero-coupon yield curve from bond market prices.

    The function solves sequentially for discount factors from shortest to
    longest maturity. For each coupon-bearing bond, all earlier coupon cash-flow
    discount factors are taken from already bootstrapped pillars or log-linearly
    interpolated from those pillars. The remaining maturity cash flow determines
    the new discount factor.

    Returns columns:
        InstrumentID, MaturityYears, DiscountFactor, ZeroRate,
        CouponRate, MarketPrice, FaceValue, Frequency
    """

    quotes = validate_bond_quotes(bonds)
    bootstrapped: dict[float, float] = {}
    rows: list[dict[str, float | str]] = []

    for _, bond in quotes.iterrows():
        instrument_id = str(bond["InstrumentID"])
        maturity = float(bond["MaturityYears"])
        coupon_rate = float(bond["CouponRate"])
        market_price = float(bond["MarketPrice"])
        face_value = float(bond["FaceValue"])
        frequency = int(bond["Frequency"])

        coupon = face_value * coupon_rate / frequency
        times = _cashflow_times(maturity, frequency)

        pv_known = 0.0
        for cashflow_time in times[:-1]:
            pv_known += coupon * _interpolated_discount_factor(cashflow_time, bootstrapped)

        final_cashflow = face_value + coupon
        discount_factor = (market_price - pv_known) / final_cashflow
        if not 0.0 < discount_factor <= 1.5:
            raise ValueError(
                f"Invalid bootstrapped discount factor {discount_factor:.8f} "
                f"for {instrument_id}; check price/coupon inputs"
            )

        zero_rate = -log(discount_factor) / maturity
        bootstrapped[maturity] = discount_factor
        rows.append(
            {
                "InstrumentID": instrument_id,
                "MaturityYears": maturity,
                "DiscountFactor": discount_factor,
                "ZeroRate": zero_rate,
                "CouponRate": coupon_rate,
                "MarketPrice": market_price,
                "FaceValue": face_value,
                "Frequency": frequency,
            }
        )

    return pd.DataFrame(rows)


class ZeroCouponYieldCurve:
    """Convenience zero-coupon curve object with interpolation helpers."""

    def __init__(self, curve: pd.DataFrame):
        required = {"MaturityYears", "DiscountFactor", "ZeroRate"}
        missing = required.difference(curve.columns)
        if missing:
            raise ValueError(f"Missing zero curve columns: {sorted(missing)}")
        self.curve = curve.sort_values("MaturityYears").reset_index(drop=True).copy()

    @classmethod
    def from_bond_quotes(cls, bonds: pd.DataFrame) -> "ZeroCouponYieldCurve":
        return cls(bootstrap_zero_coupon_curve(bonds))

    def discount_factor(self, maturity_years: float) -> float:
        pillars = dict(zip(self.curve["MaturityYears"].astype(float), self.curve["DiscountFactor"].astype(float)))
        return _interpolated_discount_factor(float(maturity_years), pillars)

    def zero_rate(self, maturity_years: float) -> float:
        t = float(maturity_years)
        if t <= 0:
            raise ValueError("maturity_years must be positive")
        return -log(self.discount_factor(t)) / t

    def forward_rate(self, start_years: float, end_years: float) -> float:
        start = float(start_years)
        end = float(end_years)
        if not 0 <= start < end:
            raise ValueError("Require 0 <= start_years < end_years")
        df_start = 1.0 if start == 0 else self.discount_factor(start)
        df_end = self.discount_factor(end)
        return log(df_start / df_end) / (end - start)

    def to_frame(self) -> pd.DataFrame:
        return self.curve.copy()


def price_zero_coupon_from_curve(face_value: float, maturity_years: float, curve: ZeroCouponYieldCurve) -> float:
    """Price a zero-coupon bond from a bootstrapped curve."""

    if face_value <= 0:
        raise ValueError("face_value must be positive")
    return float(face_value) * curve.discount_factor(float(maturity_years))


def save_sample_bond_quotes(path: str | Path) -> Path:
    """Write deterministic sample bond-market quotes to CSV."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_bond_market_quotes().to_csv(output, index=False)
    return output

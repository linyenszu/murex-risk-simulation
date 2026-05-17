from __future__ import annotations

from datetime import timedelta
import numpy as np
import pandas as pd


def generate_synthetic_positions(valuation_date: str | pd.Timestamp) -> pd.DataFrame:
    vd = pd.Timestamp(valuation_date)
    df = pd.DataFrame(
        {
            "PositionID": [f"POS{i:03d}" for i in range(1, 17)],
            "InstrumentType": [
                "Stock", "Stock", "FX Forward", "FX Forward", "European Option", "European Option",
                "Stock", "European Option", "FX Forward", "Stock", "Equity Future", "Equity Future",
                "Zero Coupon Bond", "Fixed Rate Bond", "Interest Rate Swap", "Interest Rate Swap",
            ],
            "Ticker": [
                "AAPL", "GOOG", "EURUSD=X", "GBPUSD=X", "AAPL", "GOOG", "GOOG", "AAPL",
                "EURUSD=X", "AAPL", "AAPL", "GOOG", "US10Y", "US10Y", "SOFR", "SOFR",
            ],
            "Quantity": [1000, 500, 1_000_000, -500_000, 50, -30, -200, 100, -200_000, 400, 10, -5, 250, 150, 1, -1],
            "Portfolio": [
                "P1_EqUS", "P1_EqUS", "P2_FXMaj", "P2_FXMaj", "P3_OptEq", "P3_OptEq",
                "P1_EqUS", "P4_OptSpec", "P2_FXMaj", "P4_OptSpec", "P5_Futures", "P5_Futures",
                "P6_Rates", "P6_Rates", "P7_Swaps", "P7_Swaps",
            ],
            "Maturity": [
                pd.NaT, pd.NaT, vd + timedelta(days=90), vd + timedelta(days=180),
                vd + timedelta(days=60), vd + timedelta(days=120), pd.NaT, vd + timedelta(days=90),
                vd + timedelta(days=30), pd.NaT, vd + timedelta(days=90), vd + timedelta(days=180),
                vd + timedelta(days=365 * 2), vd + timedelta(days=365 * 5), vd + timedelta(days=365 * 3), vd + timedelta(days=365 * 7),
            ],
            "Strike": [
                np.nan, np.nan, 1.08, 1.25, 170.0, 180.0, np.nan, 175.0, 1.07, np.nan,
                205.0, 155.0, np.nan, np.nan, np.nan, np.nan,
            ],
            "OptionType": [np.nan, np.nan, np.nan, np.nan, "Call", "Put", np.nan, "Call", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "ContractMultiplier": [np.nan] * 10 + [100.0, 100.0] + [np.nan] * 4,
            "FaceValue": [np.nan] * 12 + [1000.0, 1000.0] + [np.nan] * 2,
            "CouponRate": [np.nan] * 13 + [0.045] + [np.nan] * 2,
            "Notional": [np.nan] * 14 + [5_000_000.0, 3_000_000.0],
            "FixedRate": [np.nan] * 14 + [0.035, 0.032],
            "PayReceive": [np.nan] * 14 + ["Payer", "Receiver"],
            "PaymentFrequency": [np.nan] * 13 + [2.0, 2.0, 2.0],
        }
    )
    df["Maturity"] = pd.to_datetime(df["Maturity"])
    return df

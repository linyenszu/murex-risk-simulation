from __future__ import annotations

import pandas as pd


def generate_structure() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Portfolio": ["P1_EqUS", "P2_FXMaj", "P3_OptEq", "P4_OptSpec", "P5_Futures", "P6_Rates", "P7_Swaps"],
            "TradingDesk": ["Equity Desk", "FX Desk", "Options Desk", "Options Desk", "Futures Desk", "Rates Desk", "Rates Desk"],
            "Unit": ["Trading Unit A", "Trading Unit A", "Trading Unit B", "Trading Unit B", "Trading Unit A", "Trading Unit C", "Trading Unit C"],
        }
    )

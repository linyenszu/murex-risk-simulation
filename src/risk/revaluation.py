from __future__ import annotations

import numpy as np
import pandas as pd

from src.pricing.greeks import calculate_position_greeks


def revalue_portfolio(positions_df: pd.DataFrame, scenario_factor_changes: pd.Series, valuation_date, settings) -> float:
    shocked = positions_df.copy()
    shocks, base = scenario_factor_changes.align(shocked.set_index('Ticker')['CurrentPrice'], join='right')
    shocks = shocks.fillna(0.0)
    shocked_prices = base * np.exp(shocks)
    shocked['CurrentPrice'] = shocked['Ticker'].map(shocked_prices.to_dict())
    repriced = calculate_position_greeks(shocked, valuation_date, settings)
    return float(repriced['NPV'].fillna(0.0).sum())


from __future__ import annotations

import pandas as pd

from src.risk.revaluation import revalue_portfolio
from src.utils.helpers import percentile_var


def calculate_historical_var(positions_df: pd.DataFrame, historical_market_data: pd.DataFrame, valuation_date, settings):
    import numpy as np
    ratios = historical_market_data / historical_market_data.shift(1)
    factor_changes = np.log(ratios).replace([np.inf, -np.inf], 0.0).dropna()
    lookback = min(settings.var_lookback_days, len(factor_changes))
    scenarios = factor_changes.tail(lookback)
    current_npv = float(positions_df['NPV'].fillna(0.0).sum())
    pnl = []
    for _, scenario in scenarios.iterrows():
        scenario_npv = revalue_portfolio(positions_df, scenario, valuation_date, settings)
        pnl.append(scenario_npv - current_npv)
    return {
        'current_npv': current_npv,
        'scenario_count': len(scenarios),
        'pnl': pnl,
        'var_99': percentile_var(pnl, settings.confidence_level),
        'factor_changes': factor_changes,
    }

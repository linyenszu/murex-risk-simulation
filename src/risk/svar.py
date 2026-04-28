from __future__ import annotations

import pandas as pd

from src.risk.revaluation import revalue_portfolio
from src.utils.helpers import percentile_var


def select_stress_window(factor_changes: pd.DataFrame, anchor_ticker: str = 'AAPL', window: int = 252) -> pd.DataFrame:
    if factor_changes.empty:
        return factor_changes
    anchor = factor_changes[anchor_ticker] if anchor_ticker in factor_changes else factor_changes.iloc[:, 0]
    worst_day = anchor.idxmin()
    worst_loc = factor_changes.index.get_loc(worst_day)
    start_loc = max(0, worst_loc - window // 2)
    end_loc = min(len(factor_changes), start_loc + window)
    return factor_changes.iloc[start_loc:end_loc]


def calculate_stressed_var(positions_df: pd.DataFrame, factor_changes: pd.DataFrame, valuation_date, settings):
    stress_scenarios = select_stress_window(factor_changes, window=settings.var_lookback_days)
    current_npv = float(positions_df['NPV'].fillna(0.0).sum())
    pnl = []
    for _, scenario in stress_scenarios.iterrows():
        scenario_npv = revalue_portfolio(positions_df, scenario, valuation_date, settings)
        pnl.append(scenario_npv - current_npv)
    return {
        'scenario_count': len(stress_scenarios),
        'pnl': pnl,
        'svar_99': percentile_var(pnl, settings.confidence_level) if pnl else 0.0,
    }

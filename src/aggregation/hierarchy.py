from __future__ import annotations

import pandas as pd

from src.risk.svar import calculate_stressed_var
from src.risk.var import calculate_historical_var


GREEKS = ['NPV', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho']


def aggregate_greeks(positions_df: pd.DataFrame):
    desk = positions_df.groupby(['Unit', 'TradingDesk'], as_index=False)[GREEKS].sum(numeric_only=True)
    unit = positions_df.groupby(['Unit'], as_index=False)[GREEKS].sum(numeric_only=True)
    return desk, unit


def calculate_var_for_subset(positions_subset_df, historical_market_data, valuation_date, settings):
    var_result = calculate_historical_var(positions_subset_df, historical_market_data, valuation_date, settings)
    svar_result = calculate_stressed_var(positions_subset_df, var_result['factor_changes'], valuation_date, settings)
    return {
        'CurrentNPV': var_result['current_npv'],
        'VaR_99': var_result['var_99'],
        'sVaR_99': svar_result['svar_99'],
    }


def build_reports(positions_df: pd.DataFrame, historical_market_data: pd.DataFrame, valuation_date, settings):
    desk_report = []
    for (unit, desk), subset in positions_df.groupby(['Unit', 'TradingDesk']):
        metrics = calculate_var_for_subset(subset, historical_market_data, valuation_date, settings)
        metrics.update({'Unit': unit, 'TradingDesk': desk})
        desk_report.append(metrics)
    report_by_desk = pd.DataFrame(desk_report)

    unit_report = []
    for unit, subset in positions_df.groupby('Unit'):
        metrics = calculate_var_for_subset(subset, historical_market_data, valuation_date, settings)
        metrics.update({'Unit': unit})
        unit_report.append(metrics)
    report_by_unit = pd.DataFrame(unit_report)
    return report_by_desk, report_by_unit

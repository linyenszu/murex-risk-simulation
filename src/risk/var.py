from __future__ import annotations

import pandas as pd
from src.pricing.instruments import MarketContext
from src.risk.revaluation import portfolio_pnl_distribution, revalue_positions, scenario_prices


def historical_log_returns(market_data: pd.DataFrame) -> pd.DataFrame:
    return (market_data / market_data.shift(1)).apply(lambda x: __import__("numpy").log(x)).dropna()


def var_from_pnl(pnl: pd.Series, confidence_level: float) -> float:
    return float(pnl.quantile(1.0 - confidence_level))


def build_historical_scenarios(market_data: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    return historical_log_returns(market_data).tail(lookback_days)


def calculate_var(positions: pd.DataFrame, market_data: pd.DataFrame, ctx: MarketContext, confidence_level: float, lookback_days: int) -> tuple[float, pd.DataFrame]:
    scenarios = build_historical_scenarios(market_data, lookback_days)
    pnl = portfolio_pnl_distribution(positions, market_data, ctx, scenarios)
    return var_from_pnl(pnl["PortfolioPnL"], confidence_level), pnl


def grouped_pnl_distribution(positions: pd.DataFrame, market_data: pd.DataFrame, ctx: MarketContext, scenarios: pd.DataFrame, group_col: str) -> pd.DataFrame:
    base_prices = market_data.loc[: ctx.valuation_date].iloc[-1]
    rows = []
    for scenario_date, returns in scenarios.iterrows():
        shocked = scenario_prices(base_prices, returns)
        revalued = revalue_positions(positions, ctx, shocked)
        grouped = revalued.groupby(group_col)["PnL"].sum()
        rows.append(grouped.rename(scenario_date))
    return pd.DataFrame(rows).fillna(0.0)

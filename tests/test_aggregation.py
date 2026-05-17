from __future__ import annotations

import pytest

from src.aggregation.hierarchy import aggregate_greeks, hierarchy_report
from src.pricing.greeks import GREEK_COLUMNS


def test_aggregate_greeks_by_trading_desk(priced_positions) -> None:
    report = aggregate_greeks(priced_positions, ["TradingDesk"])
    assert set(report["TradingDesk"]) == set(priced_positions["TradingDesk"].unique())
    assert set(GREEK_COLUMNS).issubset(report.columns)
    assert report["NPV"].sum() == pytest.approx(priced_positions["NPV"].sum())


def test_hierarchy_report_contains_var_and_svar(priced_positions, market, market_context) -> None:
    report = hierarchy_report(
        priced_positions,
        market,
        market_context,
        confidence_level=0.99,
        lookback_days=20,
        stress_window_days=20,
        group_col="Unit",
    )
    assert set(report["Unit"]) == set(priced_positions["Unit"].unique())
    assert "VaR_99" in report.columns
    assert "sVaR_99" in report.columns

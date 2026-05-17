from __future__ import annotations

import pandas as pd

from src.data.generate_positions import generate_synthetic_positions
from src.data.generate_structure import generate_structure
from src.data.market_data import enrich_positions_with_market, get_market_data, simulate_market_data


def test_generate_synthetic_positions_schema_and_dates() -> None:
    positions = generate_synthetic_positions("2025-04-04")
    assert len(positions) == 16
    assert {"PositionID", "InstrumentType", "Ticker", "Quantity", "Portfolio", "Maturity", "Strike", "OptionType"}.issubset(positions.columns)
    assert pd.api.types.is_datetime64_any_dtype(positions["Maturity"])
    assert positions["PositionID"].is_unique
    assert {"Stock", "FX Forward", "European Option", "Equity Future", "Zero Coupon Bond", "Fixed Rate Bond", "Interest Rate Swap"}.issubset(set(positions["InstrumentType"]))


def test_generate_structure_maps_each_portfolio_once() -> None:
    structure = generate_structure()
    assert structure["Portfolio"].is_unique
    assert {"TradingDesk", "Unit"}.issubset(structure.columns)


def test_simulate_market_data_is_reproducible_and_positive() -> None:
    tickers = ("AAPL", "GOOG")
    first = simulate_market_data(tickers, "2025-04-04", years=1, seed=99)
    second = simulate_market_data(tickers, "2025-04-04", years=1, seed=99)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == list(tickers)
    assert (first > 0).all().all()
    assert first.index.max() <= pd.Timestamp("2025-04-04")


def test_get_market_data_backwards_compatible_wrapper() -> None:
    data = get_market_data(("AAPL",), historical_start_date="2020-01-01", valuation_date="2025-04-04")
    assert list(data.columns) == ["AAPL"]
    assert not data.empty


def test_enrich_positions_with_market_adds_hierarchy_price_and_stock_market_value() -> None:
    positions = generate_synthetic_positions("2025-04-04")
    structure = generate_structure()
    market = simulate_market_data(("AAPL", "GOOG", "EURUSD=X", "GBPUSD=X", "US10Y", "SOFR"), "2025-04-04", years=1, seed=1)
    enriched = enrich_positions_with_market(positions, structure, market, "2025-04-04")
    assert enriched["TradingDesk"].notna().all()
    assert enriched["Unit"].notna().all()
    assert enriched["CurrentPrice"].notna().all()
    stock_rows = enriched["InstrumentType"].eq("Stock")
    assert enriched.loc[stock_rows, "MarketValue_Initial"].notna().all()
    assert enriched.loc[~stock_rows, "MarketValue_Initial"].isna().all()

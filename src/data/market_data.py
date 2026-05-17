from __future__ import annotations

from datetime import timedelta
import numpy as np
import pandas as pd


def simulate_market_data(tickers: tuple[str, ...], valuation_date: str, years: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(valuation_date)
    dates = pd.bdate_range(end=end, periods=years * 252)
    start_prices = {"AAPL": 170.0, "GOOG": 140.0, "EURUSD=X": 1.08, "GBPUSD=X": 1.25, "US10Y": 0.04, "SOFR": 0.035}
    vols = {"AAPL": 0.022, "GOOG": 0.024, "EURUSD=X": 0.0045, "GBPUSD=X": 0.0055, "US10Y": 0.015, "SOFR": 0.01}
    drifts = {"AAPL": 0.0004, "GOOG": 0.00035, "EURUSD=X": 0.00002, "GBPUSD=X": 0.00001, "US10Y": 0.0, "SOFR": 0.0}
    data: dict[str, np.ndarray] = {}
    for ticker in tickers:
        returns = rng.normal(drifts.get(ticker, 0.0001), vols.get(ticker, 0.01), len(dates))
        data[ticker] = start_prices.get(ticker, 100.0) * np.exp(np.cumsum(returns))
    return pd.DataFrame(data, index=dates).round(6)


def load_market_data(tickers: tuple[str, ...], valuation_date: str, seed: int = 42, use_yfinance: bool = False) -> pd.DataFrame:
    if use_yfinance:
        try:
            import yfinance as yf
            end = pd.Timestamp(valuation_date)
            start = end - timedelta(days=5 * 365)
            raw = yf.download(list(tickers), start=start, end=end + timedelta(days=1), progress=False, auto_adjust=False)
            close = raw["Close"].ffill().bfill()
            if not close.empty:
                return close.loc[:valuation_date]
        except Exception:
            pass
    return simulate_market_data(tickers, valuation_date, seed=seed)


def enrich_positions_with_market(positions: pd.DataFrame, structure: pd.DataFrame, market: pd.DataFrame, valuation_date: str) -> pd.DataFrame:
    latest = market.loc[:valuation_date].iloc[-1].rename("CurrentPrice").reset_index().rename(columns={"index": "Ticker"})
    out = positions.merge(structure, on="Portfolio", how="left").merge(latest, on="Ticker", how="left")
    out["MarketValue_Initial"] = np.where(out["InstrumentType"].eq("Stock"), out["Quantity"] * out["CurrentPrice"], np.nan)
    return out


def get_market_data(
    tickers: tuple[str, ...],
    historical_start_date: str | None = None,
    valuation_date: str = "2025-04-04",
    seed: int = 42,
    use_yfinance: bool = False,
) -> pd.DataFrame:
    """Backward-compatible wrapper for notebooks/scripts.

    ``historical_start_date`` is accepted for the original API. The synthetic
    data path derives a five-year business-day history ending on valuation_date;
    the yfinance path uses load_market_data's standard five-year window.
    """
    return load_market_data(tickers, valuation_date, seed=seed, use_yfinance=use_yfinance)

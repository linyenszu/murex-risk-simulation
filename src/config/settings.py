from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RiskSettings:
    valuation_date: str = "2025-04-04"
    confidence_level: float = 0.99
    lookback_days: int = 252
    stress_window_days: int = 252
    random_seed: int = 42
    base_currency: str = "USD"
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    tickers: tuple[str, ...] = ("AAPL", "GOOG", "EURUSD=X", "GBPUSD=X", "US10Y", "SOFR")
    risk_free_rate: float = 0.02
    dividend_yield: float = 0.00
    ticker_vols: dict[str, float] | None = None

    def vols(self) -> dict[str, float]:
        return self.ticker_vols or {
            "AAPL": 0.20,
            "GOOG": 0.25,
            "EURUSD=X": 0.15,
            "GBPUSD=X": 0.12,
            "US10Y": 0.01,
            "SOFR": 0.005,
        }

    def validate(self) -> None:
        if not 0.90 <= self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0.90 and 1.0")
        if self.lookback_days < 30:
            raise ValueError("lookback_days must be at least 30")


@dataclass(frozen=True)
class AppConfig(RiskSettings):
    """Backward-compatible application config used by scripts/notebooks.

    The project originally referenced AppConfig with market_tickers and
    historical_start_date attributes. RiskSettings is now the canonical config,
    but this alias keeps older scripts and notebooks working.
    """

    @property
    def market_tickers(self) -> tuple[str, ...]:
        return self.tickers

    @property
    def historical_start_date(self) -> str:
        import pandas as pd

        return (pd.Timestamp(self.valuation_date) - pd.Timedelta(days=5 * 365)).strftime("%Y-%m-%d")

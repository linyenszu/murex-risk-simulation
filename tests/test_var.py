import pandas as pd

from src.config.settings import Settings
from src.pricing.greeks import calculate_position_greeks
from src.risk.var import calculate_historical_var


def test_historical_var_returns_expected_keys():
    settings = Settings()
    valuation_date = settings.valuation_date
    positions = pd.DataFrame([
        {'InstrumentType': 'Stock', 'Ticker': 'AAPL', 'Quantity': 10, 'Portfolio': 'P1', 'Maturity': pd.NaT, 'Strike': None, 'OptionType': None, 'CurrentPrice': 100.0, 'TradingDesk': 'Equity Desk', 'Unit': 'Trading Unit A'}
    ])
    greeks = calculate_position_greeks(positions, valuation_date, settings)
    market = pd.DataFrame({'AAPL': [98.0, 99.0, 100.0, 101.0, 100.0]}, index=pd.to_datetime(['2025-03-31','2025-04-01','2025-04-02','2025-04-03','2025-04-04']))
    result = calculate_historical_var(greeks, market, valuation_date, settings)
    assert 'var_99' in result
    assert 'pnl' in result
    assert result['scenario_count'] > 0

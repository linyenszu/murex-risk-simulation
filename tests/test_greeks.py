import pandas as pd

from src.config.settings import Settings
from src.data.generate_positions import generate_synthetic_positions
from src.data.generate_structure import generate_structure
from src.pricing.greeks import calculate_position_greeks


def test_calculate_position_greeks_has_expected_columns():
    settings = Settings()
    valuation_date = settings.valuation_date
    positions = generate_synthetic_positions(valuation_date)
    structure = generate_structure()
    merged = positions.merge(structure, on='Portfolio', how='left')
    price_map = {'AAPL': 203.19, 'GOOG': 152.63, 'EURUSD=X': 1.0989, 'GBPUSD=X': 1.2990}
    merged['CurrentPrice'] = merged['Ticker'].map(price_map)

    output = calculate_position_greeks(merged, valuation_date, settings)
    for col in ['NPV', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho']:
        assert col in output.columns
    assert output['NPV'].notna().sum() >= 8


def test_stock_delta_matches_quantity():
    settings = Settings()
    valuation_date = settings.valuation_date
    df = pd.DataFrame([
        {'InstrumentType': 'Stock', 'Ticker': 'AAPL', 'Quantity': 10, 'Portfolio': 'P1_EqUS', 'Maturity': pd.NaT, 'Strike': None, 'OptionType': None, 'CurrentPrice': 100.0}
    ])
    output = calculate_position_greeks(df, valuation_date, settings)
    assert output.loc[0, 'Delta'] == 10
    assert output.loc[0, 'NPV'] == 1000.0

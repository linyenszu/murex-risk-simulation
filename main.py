from __future__ import annotations

import pandas as pd

from src.aggregation.hierarchy import build_reports
from src.config.settings import Settings
from src.data.generate_positions import generate_synthetic_positions
from src.data.generate_structure import generate_structure
from src.data.market_data import fetch_historical_market_data, get_latest_prices
from src.pricing.greeks import calculate_position_greeks
from src.risk.svar import calculate_stressed_var
from src.risk.var import calculate_historical_var
from src.utils.helpers import ensure_parent
from src.utils.logger import get_logger


logger = get_logger(__name__)


def main() -> None:
    settings = Settings()
    valuation_date = settings.valuation_date

    logger.info('Generating synthetic positions and structure data')
    positions_df = generate_synthetic_positions(valuation_date)
    structure_df = generate_structure()

    logger.info('Fetching historical market data')
    market_data_df = fetch_historical_market_data(
        settings.market_tickers,
        settings.historical_start_date,
        settings.historical_end_date,
    )

    logger.info('Saving raw inputs')
    ensure_parent(settings.positions_filename)
    positions_df.to_csv(settings.positions_filename, index=False)
    structure_df.to_csv(settings.structure_filename, index=False)
    market_data_df.to_csv(settings.market_data_filename)

    merged = positions_df.merge(structure_df, on='Portfolio', how='left')
    latest_prices_df = get_latest_prices(market_data_df, settings.valuation_date_str)
    merged = merged.merge(latest_prices_df, on='Ticker', how='left')
    merged['MarketValue_Initial'] = merged['Quantity'] * merged['CurrentPrice']
    merged.loc[merged['InstrumentType'] != 'Stock', 'MarketValue_Initial'] = pd.NA

    logger.info('Calculating instrument-level Greeks and NPV')
    position_greeks = calculate_position_greeks(merged, valuation_date, settings)
    position_greeks.to_csv(ensure_parent(settings.greeks_output_filename), index=False)

    logger.info('Calculating portfolio VaR and stressed VaR')
    var_result = calculate_historical_var(position_greeks, market_data_df, valuation_date, settings)
    svar_result = calculate_stressed_var(position_greeks, var_result['factor_changes'], valuation_date, settings)
    portfolio_var = pd.DataFrame([
        {
            'CurrentNPV': var_result['current_npv'],
            'ScenarioCount': var_result['scenario_count'],
            'VaR_99': var_result['var_99'],
            'sVaR_99': svar_result['svar_99'],
        }
    ])
    portfolio_var.to_csv(ensure_parent(settings.portfolio_var_filename), index=False)

    logger.info('Building hierarchy reports')
    report_by_desk, report_by_unit = build_reports(position_greeks, market_data_df, valuation_date, settings)
    report_by_desk.to_csv(ensure_parent(settings.report_by_desk_filename), index=False)
    report_by_unit.to_csv(ensure_parent(settings.report_by_unit_filename), index=False)

    logger.info('Done')
    logger.info('Portfolio summary\n%s', portfolio_var.to_string(index=False))
    logger.info('Desk report\n%s', report_by_desk.to_string(index=False))
    logger.info('Unit report\n%s', report_by_unit.to_string(index=False))


if __name__ == '__main__':
    main()

from __future__ import annotations

from pathlib import Path

import pandas as pd

from main import build_market_context, price_options_with_engines, run
from src.config.settings import RiskSettings
from src.data.generate_positions import generate_synthetic_positions
from src.data.generate_structure import generate_structure
from src.data.market_data import enrich_positions_with_market, simulate_market_data
from src.pricing.greeks import calculate_instrument_risk


def _option_test_frame(tmp_path: Path) -> tuple[pd.DataFrame, RiskSettings]:
    settings = RiskSettings(
        lookback_days=30,
        stress_window_days=30,
        processed_data_dir=tmp_path / "processed",
        random_seed=123,
    )
    positions = generate_synthetic_positions(settings.valuation_date)
    structure = generate_structure()
    market = simulate_market_data(settings.tickers, settings.valuation_date, years=1, seed=settings.random_seed)
    ctx = build_market_context(settings)
    enriched = enrich_positions_with_market(positions, structure, market, settings.valuation_date)
    instrument_risk = calculate_instrument_risk(enriched, ctx)
    return instrument_risk, settings


def test_price_options_with_both_engines_contains_model_comparison(tmp_path: Path) -> None:
    instrument_risk, settings = _option_test_frame(tmp_path)
    ctx = build_market_context(settings)

    comparison = price_options_with_engines(
        instrument_risk,
        ctx,
        option_pricing_engine="both",
        mc_paths=25_000,
        mc_seed=99,
    )

    assert not comparison.empty
    assert comparison["BlackScholesUnitPrice"].notna().all()
    assert comparison["MonteCarloUnitPrice"].notna().all()
    assert comparison["MCMinusBSUnitPrice"].notna().all()
    assert (comparison["MonteCarloStdError"] > 0.0).all()
    assert set(comparison["OptionType"]).issubset({"Call", "Put"})


def test_price_options_with_black_scholes_only_leaves_mc_columns_empty(tmp_path: Path) -> None:
    instrument_risk, settings = _option_test_frame(tmp_path)
    ctx = build_market_context(settings)

    comparison = price_options_with_engines(
        instrument_risk,
        ctx,
        option_pricing_engine="black-scholes",
        mc_paths=1_000,
        mc_seed=1,
    )

    assert comparison["BlackScholesUnitPrice"].notna().all()
    assert comparison["MonteCarloUnitPrice"].isna().all()
    assert comparison["MCMinusBSUnitPrice"].isna().all()


def test_main_run_writes_option_pricing_comparison(tmp_path: Path) -> None:
    settings = RiskSettings(
        lookback_days=30,
        stress_window_days=30,
        processed_data_dir=tmp_path / "processed",
        random_seed=77,
    )

    result = run(settings, option_pricing_engine="both", mc_paths=10_000, mc_seed=77)

    output_file = tmp_path / "processed" / "option_pricing_comparison.csv"
    assert output_file.exists()
    assert "option_pricing" in result
    written = pd.read_csv(output_file)
    assert not written.empty
    assert "MonteCarloUnitPrice" in written.columns

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from main import run
from src.config.settings import RiskSettings
from src.data.generate_positions import generate_synthetic_positions
from src.data.generate_structure import generate_structure
from src.data.loaders import enrich_positions, load_market_data as load_market_csv, load_positions, load_structure
from src.data.market_data import simulate_market_data
from src.utils.helpers import ensure_dir, require_columns


def test_loaders_roundtrip_and_enrichment(tmp_path: Path) -> None:
    positions = generate_synthetic_positions("2025-04-04")
    structure = generate_structure()
    market = simulate_market_data(("AAPL", "GOOG", "EURUSD=X", "GBPUSD=X", "US10Y", "SOFR"), "2025-04-04", years=1, seed=5)

    positions_path = tmp_path / "positions.csv"
    structure_path = tmp_path / "structure.csv"
    market_path = tmp_path / "market.csv"
    positions.to_csv(positions_path, index=False)
    structure.to_csv(structure_path, index=False)
    market.to_csv(market_path, index_label="Date")

    loaded_positions = load_positions(positions_path)
    loaded_structure = load_structure(structure_path)
    loaded_market = load_market_csv(market_path)
    enriched = enrich_positions(loaded_positions, loaded_structure, loaded_market, "2025-04-04")

    assert len(enriched) == len(positions)
    assert enriched["CurrentPrice"].notna().all()
    assert enriched.loc[enriched["InstrumentType"].eq("Stock"), "MarketValue_Initial"].notna().all()


def test_require_columns_raises_with_missing_column() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        require_columns(pd.DataFrame({"A": [1]}), ["A", "B"], "frame")


def test_ensure_dir_creates_nested_directory(tmp_path: Path) -> None:
    path = ensure_dir(tmp_path / "a" / "b")
    assert path.exists()
    assert path.is_dir()


def test_main_run_writes_expected_outputs_to_configured_directory(tmp_path: Path) -> None:
    settings = RiskSettings(
        lookback_days=30,
        stress_window_days=30,
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        random_seed=11,
    )
    result = run(settings)
    assert isinstance(result["portfolio_var"], float)
    for name in [
        "positions_enriched.csv",
        "instrument_risk.csv",
        "portfolio_pnl.csv",
        "risk_report_by_desk.csv",
        "risk_report_by_unit.csv",
    ]:
        assert (tmp_path / "processed" / name).exists()

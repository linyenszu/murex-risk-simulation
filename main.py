from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd

from src.aggregation.hierarchy import hierarchy_report
from src.config.settings import RiskSettings
from src.data.generate_positions import generate_synthetic_positions
from src.data.generate_structure import generate_structure
from src.data.market_data import enrich_positions_with_market, load_market_data
from src.pricing.fallback_bs import black_scholes_greeks
from src.pricing.greeks import calculate_instrument_risk
from src.pricing.instruments import MarketContext
from src.pricing.monte_carlo import price_european_option_mc
from src.pricing.lattice_models import (
    price_european_option_binomial,
    price_european_option_trinomial,
)
from src.pricing.yield_curve import ZeroCouponYieldCurve, bootstrap_zero_coupon_curve
from src.risk.var import calculate_var
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

OptionPricingEngine = Literal["black-scholes", "monte-carlo", "binomial", "trinomial", "both", "all"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Murex-like risk simulation pipeline")
    parser.add_argument("--valuation-date", default="2025-04-04")
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--lookback-days", type=int, default=252)
    parser.add_argument("--use-yfinance", action="store_true")
    parser.add_argument(
        "--option-pricing-engine",
        choices=["black-scholes", "monte-carlo", "binomial", "trinomial", "both", "all"],
        default="all",
        help=(
            "Pricing engine used for European option reporting. "
            "Risk Greeks remain analytic Black-Scholes; this controls the extra option pricing output."
        ),
    )
    parser.add_argument(
        "--mc-paths",
        type=int,
        default=100_000,
        help="Number of Monte Carlo paths for European option pricing when enabled.",
    )
    parser.add_argument(
        "--mc-seed",
        type=int,
        default=42,
        help="Random seed for deterministic Monte Carlo option pricing.",
    )
    parser.add_argument(
        "--tree-steps",
        type=int,
        default=200,
        help="Number of risk-neutral GBM lattice steps for binomial/trinomial engines.",
    )
    parser.add_argument(
        "--bond-quotes-path",
        default="data/raw/bond_market_quotes.csv",
        help="CSV file with bond quotes used to bootstrap the zero-coupon discount curve.",
    )
    parser.add_argument(
        "--no-yield-curve",
        action="store_true",
        help="Use flat risk_free_rate instead of bootstrapped zero-coupon curve.",
    )
    return parser.parse_args()


def load_zero_curve(bond_quotes_path: str | Path | None) -> ZeroCouponYieldCurve | None:
    """Load and bootstrap a zero-coupon curve from bond-market quotes."""
    if bond_quotes_path is None:
        return None
    path = Path(bond_quotes_path)
    if not path.exists():
        return None
    bond_quotes = pd.read_csv(path)
    return ZeroCouponYieldCurve(bootstrap_zero_coupon_curve(bond_quotes))


def build_market_context(settings: RiskSettings, zero_curve: ZeroCouponYieldCurve | None = None) -> MarketContext:
    """Create the market context shared by pricing, revaluation, and VaR."""
    return MarketContext(
        valuation_date=pd.Timestamp(settings.valuation_date),
        risk_free_rate=settings.risk_free_rate,
        dividend_yield=settings.dividend_yield,
        vols=settings.vols(),
        fx_foreign_rates={"EURUSD=X": 0.015, "GBPUSD=X": 0.018},
        zero_curve=zero_curve,
    )


def _resolve_option_engines(option_pricing_engine: OptionPricingEngine) -> list[str]:
    """Resolve CLI aliases into concrete option pricing engines."""
    engine_map = {
        "black-scholes": ["black-scholes"],
        "monte-carlo": ["monte-carlo"],
        "binomial": ["binomial"],
        "trinomial": ["trinomial"],
        "both": ["black-scholes", "monte-carlo"],
        "all": ["black-scholes", "monte-carlo", "binomial", "trinomial"],
    }
    try:
        return engine_map[option_pricing_engine]
    except KeyError as exc:
        raise ValueError(f"Unsupported option pricing engine: {option_pricing_engine}") from exc


def price_options_with_engines(
    positions: pd.DataFrame,
    ctx: MarketContext,
    option_pricing_engine: OptionPricingEngine = "all",
    mc_paths: int = 100_000,
    mc_seed: int = 42,
    tree_steps: int = 200,
) -> pd.DataFrame:
    """Price European option rows with selected engines.

    Supported engines:
    * ``black-scholes``: analytic benchmark.
    * ``monte-carlo``: risk-neutral GBM terminal simulation.
    * ``binomial``: CRR risk-neutral GBM recombining tree.
    * ``trinomial``: moment-matched risk-neutral GBM recombining tree.
    * ``both``: Black-Scholes + Monte Carlo.
    * ``all``: all four engines.

    The output is a model-comparison table suitable for convergence diagnostics,
    validation checks, and front-office model explainability.
    """
    engines = _resolve_option_engines(option_pricing_engine)
    if "monte-carlo" in engines and mc_paths < 2:
        raise ValueError("mc_paths must be at least 2")
    if any(engine in engines for engine in {"binomial", "trinomial"}) and tree_steps < 1:
        raise ValueError("tree_steps must be at least 1")

    option_rows = positions[positions["InstrumentType"].eq("European Option")].copy()
    columns = [
        "PositionID",
        "Ticker",
        "OptionType",
        "Quantity",
        "CurrentPrice",
        "Strike",
        "MaturityYears",
        "Volatility",
        "BlackScholesUnitPrice",
        "BlackScholesPositionValue",
        "MonteCarloUnitPrice",
        "MonteCarloPositionValue",
        "MonteCarloStdError",
        "MonteCarloCILow",
        "MonteCarloCIHigh",
        "BinomialUnitPrice",
        "BinomialPositionValue",
        "TrinomialUnitPrice",
        "TrinomialPositionValue",
        "MCMinusBSUnitPrice",
        "BinomialMinusBSUnitPrice",
        "TrinomialMinusBSUnitPrice",
        "MCMinusBSPositionValue",
        "BinomialMinusBSPositionValue",
        "TrinomialMinusBSPositionValue",
        "MonteCarloPaths",
        "MonteCarloSeed",
        "TreeSteps",
    ]
    if option_rows.empty:
        return pd.DataFrame(columns=columns)

    results: list[dict[str, float | int | str | None]] = []
    for _, row in option_rows.iterrows():
        spot = float(row["CurrentPrice"])
        strike = float(row["Strike"])
        maturity_years = ctx.time_to_maturity(row["Maturity"])
        quantity = float(row["Quantity"])
        ticker = str(row["Ticker"])
        option_type = str(row["OptionType"])
        volatility = float(ctx.vols.get(ticker, 0.20))

        bs_unit_price: float | None = None
        bs_position_value: float | None = None
        if "black-scholes" in engines:
            bs = black_scholes_greeks(
                spot=spot,
                strike=strike,
                maturity_years=maturity_years,
                rate=ctx.zero_rate(maturity_years),
                volatility=volatility,
                option_type=option_type,
                dividend_yield=ctx.dividend_yield,
            )
            bs_unit_price = bs.price
            bs_position_value = bs.price * quantity

        mc_unit_price: float | None = None
        mc_position_value: float | None = None
        mc_std_error: float | None = None
        mc_ci_low: float | None = None
        mc_ci_high: float | None = None
        if "monte-carlo" in engines:
            seeded_stream = mc_seed + int(row.name)
            mc = price_european_option_mc(
                spot=spot,
                strike=strike,
                maturity_years=maturity_years,
                rate=ctx.zero_rate(maturity_years),
                volatility=volatility,
                option_type=option_type,
                num_paths=mc_paths,
                dividend_yield=ctx.dividend_yield,
                seed=seeded_stream,
                antithetic=True,
            )
            mc_unit_price = mc.price
            mc_position_value = mc.price * quantity
            mc_std_error = mc.standard_error
            mc_ci_low = mc.confidence_interval_low
            mc_ci_high = mc.confidence_interval_high

        binomial_unit_price: float | None = None
        binomial_position_value: float | None = None
        if "binomial" in engines:
            binomial = price_european_option_binomial(
                spot=spot,
                strike=strike,
                maturity_years=maturity_years,
                rate=ctx.zero_rate(maturity_years),
                volatility=volatility,
                option_type=option_type,
                dividend_yield=ctx.dividend_yield,
                steps=tree_steps,
            )
            binomial_unit_price = binomial.price
            binomial_position_value = binomial.price * quantity

        trinomial_unit_price: float | None = None
        trinomial_position_value: float | None = None
        if "trinomial" in engines:
            trinomial = price_european_option_trinomial(
                spot=spot,
                strike=strike,
                maturity_years=maturity_years,
                rate=ctx.zero_rate(maturity_years),
                volatility=volatility,
                option_type=option_type,
                dividend_yield=ctx.dividend_yield,
                steps=tree_steps,
            )
            trinomial_unit_price = trinomial.price
            trinomial_position_value = trinomial.price * quantity

        def _diff(model_price: float | None, benchmark: float | None) -> float | None:
            if model_price is None or benchmark is None:
                return None
            return model_price - benchmark

        results.append(
            {
                "PositionID": row["PositionID"],
                "Ticker": ticker,
                "OptionType": option_type,
                "Quantity": quantity,
                "CurrentPrice": spot,
                "Strike": strike,
                "MaturityYears": maturity_years,
                "Volatility": volatility,
                "BlackScholesUnitPrice": bs_unit_price,
                "BlackScholesPositionValue": bs_position_value,
                "MonteCarloUnitPrice": mc_unit_price,
                "MonteCarloPositionValue": mc_position_value,
                "MonteCarloStdError": mc_std_error,
                "MonteCarloCILow": mc_ci_low,
                "MonteCarloCIHigh": mc_ci_high,
                "BinomialUnitPrice": binomial_unit_price,
                "BinomialPositionValue": binomial_position_value,
                "TrinomialUnitPrice": trinomial_unit_price,
                "TrinomialPositionValue": trinomial_position_value,
                "MCMinusBSUnitPrice": _diff(mc_unit_price, bs_unit_price),
                "BinomialMinusBSUnitPrice": _diff(binomial_unit_price, bs_unit_price),
                "TrinomialMinusBSUnitPrice": _diff(trinomial_unit_price, bs_unit_price),
                "MCMinusBSPositionValue": _diff(mc_position_value, bs_position_value),
                "BinomialMinusBSPositionValue": _diff(binomial_position_value, bs_position_value),
                "TrinomialMinusBSPositionValue": _diff(trinomial_position_value, bs_position_value),
                "MonteCarloPaths": mc_paths if "monte-carlo" in engines else None,
                "MonteCarloSeed": mc_seed if "monte-carlo" in engines else None,
                "TreeSteps": tree_steps if any(engine in engines for engine in {"binomial", "trinomial"}) else None,
            }
        )

    return pd.DataFrame(results, columns=columns)

def run(
    settings: RiskSettings,
    option_pricing_engine: OptionPricingEngine = "both",
    mc_paths: int = 100_000,
    mc_seed: int | None = None,
    tree_steps: int = 200,
    bond_quotes_path: str | Path | None = "data/raw/bond_market_quotes.csv",
    use_yield_curve: bool = True,
) -> dict[str, pd.DataFrame | float]:
    settings.validate()
    log = get_logger()
    output_dir = ensure_dir(settings.processed_data_dir)
    resolved_mc_seed = settings.random_seed if mc_seed is None else mc_seed

    log.info("Generating positions and hierarchy")
    positions = generate_synthetic_positions(settings.valuation_date)
    structure = generate_structure()

    log.info("Loading market data")
    market = load_market_data(settings.tickers, settings.valuation_date, seed=settings.random_seed)
    enriched = enrich_positions_with_market(positions, structure, market, settings.valuation_date)

    zero_curve = load_zero_curve(bond_quotes_path) if use_yield_curve else None
    if zero_curve is not None:
        log.info("Bootstrapped zero-coupon curve from %s", bond_quotes_path)
        zero_curve.to_frame().to_csv(output_dir / "zero_coupon_curve.csv", index=False)
    else:
        log.info("Using flat risk-free rate %.4f", settings.risk_free_rate)

    ctx = build_market_context(settings, zero_curve=zero_curve)

    log.info("Pricing instruments and calculating Greeks")
    instrument_risk = calculate_instrument_risk(enriched, ctx)

    log.info("Pricing European options with %s engine", option_pricing_engine)
    option_pricing = price_options_with_engines(
        instrument_risk,
        ctx,
        option_pricing_engine=option_pricing_engine,
        mc_paths=mc_paths,
        mc_seed=resolved_mc_seed,
        tree_steps=tree_steps,
    )

    log.info("Calculating portfolio Historical VaR")
    portfolio_var, pnl = calculate_var(
        instrument_risk,
        market,
        ctx,
        confidence_level=settings.confidence_level,
        lookback_days=settings.lookback_days,
    )

    log.info("Building hierarchy reports")
    desk_report = hierarchy_report(
        instrument_risk,
        market,
        ctx,
        settings.confidence_level,
        settings.lookback_days,
        settings.stress_window_days,
        group_col="TradingDesk",
    )
    unit_report = hierarchy_report(
        instrument_risk,
        market,
        ctx,
        settings.confidence_level,
        settings.lookback_days,
        settings.stress_window_days,
        group_col="Unit",
    )

    enriched.to_csv(output_dir / "positions_enriched.csv", index=False)
    instrument_risk.to_csv(output_dir / "instrument_risk.csv", index=False)
    option_pricing.to_csv(output_dir / "option_pricing_comparison.csv", index=False)
    pnl.to_csv(output_dir / "portfolio_pnl.csv")
    desk_report.to_csv(output_dir / "risk_report_by_desk.csv", index=False)
    unit_report.to_csv(output_dir / "risk_report_by_unit.csv", index=False)

    log.info("Portfolio VaR %.2f", portfolio_var)
    log.info("Wrote outputs to %s", Path(output_dir).resolve())
    return {
        "portfolio_var": portfolio_var,
        "instrument_risk": instrument_risk,
        "option_pricing": option_pricing,
        "portfolio_pnl": pnl,
        "desk_report": desk_report,
        "unit_report": unit_report,
        "zero_curve": zero_curve.to_frame() if zero_curve is not None else pd.DataFrame(),
    }


if __name__ == "__main__":
    args = parse_args()
    run(
        RiskSettings(
            valuation_date=args.valuation_date,
            confidence_level=args.confidence,
            lookback_days=args.lookback_days,
        ),
        option_pricing_engine=args.option_pricing_engine,
        mc_paths=args.mc_paths,
        mc_seed=args.mc_seed,
        tree_steps=args.tree_steps,
        bond_quotes_path=args.bond_quotes_path,
        use_yield_curve=not args.no_yield_curve,
    )

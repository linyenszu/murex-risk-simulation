"""Run Monte Carlo European option pricing and convergence analysis."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pricing.fallback_bs import black_scholes_greeks
from src.pricing.monte_carlo import convergence_analysis, price_european_option_mc


def main() -> None:
    spot = 100.0
    strike = 100.0
    maturity_years = 1.0
    rate = 0.05
    volatility = 0.20
    dividend_yield = 0.0
    option_type = "Call"

    result = price_european_option_mc(
        spot=spot,
        strike=strike,
        maturity_years=maturity_years,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        option_type=option_type,
        num_paths=100_000,
        seed=42,
        antithetic=True,
    )
    bs = black_scholes_greeks(spot, strike, maturity_years, rate, volatility, option_type, dividend_yield)
    convergence = convergence_analysis(
        spot=spot,
        strike=strike,
        maturity_years=maturity_years,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        option_type=option_type,
        seed=42,
    )

    output_dir = PROJECT_ROOT / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    convergence_path = output_dir / "mc_option_convergence.csv"
    convergence.to_csv(convergence_path, index=False)

    print("Monte Carlo European Option Pricing")
    print(f"MC price: {result.price:.6f}")
    print(f"95% CI: [{result.confidence_interval_low:.6f}, {result.confidence_interval_high:.6f}]")
    print(f"Black-Scholes price: {bs.price:.6f}")
    print(f"Absolute error: {abs(result.price - bs.price):.6f}")
    print(f"Convergence table written to: {convergence_path}")


if __name__ == "__main__":
    main()

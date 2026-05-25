# Py Risk Simulation

Production-style Python repository that simulates a Murex-like front-office risk pipeline for positions, market data, Greeks, Historical Simulation VaR, stressed VaR, and desk/unit aggregation.

## Capabilities

- Generate synthetic multi-asset positions: stocks, equity futures, FX forwards, European options, zero-coupon bonds, fixed-rate bonds, and vanilla interest-rate swaps
- Generate portfolio-to-desk-to-unit hierarchy data
- Load market data with a deterministic fallback for offline execution
- Price instruments and calculate Greeks / rate sensitivities across equity, FX, options, futures, rates, and swaps
- Revalue positions under historical scenarios
- Compute VaR and stressed VaR
- Aggregate risk by portfolio, trading desk, and business unit
- Run tests and pipeline from CLI, Makefile, Docker, or notebooks

## Repository Layout

See `Repository_Structure.txt` for the requested structure this implementation follows.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
pytest
```

## Pipeline

```bash
python scripts/bootstrap_data.py
python main.py
```

Option trades can now be priced from `main.py` with multiple engines:

```bash
# Default: write Black-Scholes vs Monte Carlo comparison
python main.py --option-pricing-engine both --mc-paths 100000

# Analytic option prices only
python main.py --option-pricing-engine black-scholes

# Monte Carlo option prices only
python main.py --option-pricing-engine monte-carlo --mc-paths 250000 --mc-seed 7
```

Outputs are written to `data/processed/` and `data/outputs/`. The main pipeline now writes `data/processed/option_pricing_comparison.csv` with Black-Scholes prices, Monte Carlo prices, standard errors, confidence intervals, and MC-minus-BS differences for European option trades.

## Notes

QuantLib-Python is optional. If it is unavailable, the project uses pure-Python pricing functions for European options, futures, bonds, FX forwards, and swaps so the repo remains runnable in lightweight environments.

## Supported Trade Types

| InstrumentType | Pricing approach | Primary risk driver |
| --- | --- | --- |
| `Stock` | Linear spot valuation | Equity spot |
| `Equity Future` | Discounted futures payoff | Equity spot / futures strike |
| `FX Forward` | Interest-rate parity forward valuation | FX spot and foreign/domestic rates |
| `European Option` | Black-Scholes price and Greeks | Spot, strike, vol, time, rates |
| `Zero Coupon Bond` | Discounted cash flow | Yield/rate shock |
| `Fixed Rate Bond` | Coupon bond discounted cash flow | Yield/rate shock |
| `Interest Rate Swap` | Par-rate spread times annuity approximation | SOFR/rate shock |

## Pricing Models

### Black-Scholes Analytical Model

Supports:
- European Calls
- European Puts
- Greeks:
  - Delta
  - Gamma
  - Vega
  - Theta
  - Rho

---

## Monte Carlo Option Pricing

This version adds a production-style Monte Carlo engine for European options under risk-neutral Geometric Brownian Motion.

Key concepts implemented:

- Geometric Brownian Motion terminal-price simulation
- Random path generation for diagnostics and future path-dependent products
- Risk-neutral discounted payoff valuation
- Antithetic variates for variance reduction
- Confidence intervals and standard errors
- Convergence analysis across path counts
- Black-Scholes benchmark comparison

Run the demo:

```bash
python scripts/run_mc_option_pricing.py
```

The convergence table is written to:

```text
data/outputs/mc_option_convergence.csv
```

Core API:

```python
from src.pricing.monte_carlo import price_european_option_mc, convergence_analysis

result = price_european_option_mc(
    spot=100,
    strike=100,
    maturity_years=1.0,
    rate=0.05,
    volatility=0.20,
    option_type="Call",
    num_paths=100_000,
    seed=42,
)

convergence = convergence_analysis(
    spot=100,
    strike=100,
    maturity_years=1.0,
    rate=0.05,
    volatility=0.20,
    option_type="Call",
)

```

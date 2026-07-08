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

---

## Zero-Coupon Yield Curve Bootstrapping

The repository now includes a rates-market bootstrapping component that constructs a continuously compounded zero-coupon yield curve from bond market quotes.

Core module:

```text
src/pricing/yield_curve.py
```

Sample market data:

```text
data/raw/bond_market_quotes.csv
```

Generated curve output:

```text
data/processed/zero_coupon_curve.csv
```

Run the curve construction script:

```bash
python scripts/bootstrap_yield_curve.py --write-sample-input
```

Or provide your own bond quote file:

```bash
python scripts/bootstrap_yield_curve.py \
  --input data/raw/bond_market_quotes.csv \
  --output data/processed/zero_coupon_curve.csv
```

Expected input columns:

| Column | Description |
| --- | --- |
| `InstrumentID` | Bond identifier |
| `MaturityYears` | Time to maturity in years |
| `CouponRate` | Annual coupon rate, e.g. `0.04` for 4% |
| `MarketPrice` | Dirty price per face value, usually per 100 notional |
| `FaceValue` | Optional, defaults to `100` |
| `Frequency` | Optional coupon frequency, defaults to semiannual `2` |

The bootstrapping algorithm:

1. Sorts bonds by maturity.
2. Solves the shortest zero-coupon pillars directly from market price.
3. Discounts earlier coupon cash flows using already bootstrapped discount factors.
4. Solves the remaining maturity cash flow for the next discount factor.
5. Converts discount factors into continuously compounded zero rates.
6. Uses log-linear discount-factor interpolation for off-pillar zero and forward rates.

Example API:

```python
import pandas as pd
from src.pricing.yield_curve import ZeroCouponYieldCurve, bootstrap_zero_coupon_curve

quotes = pd.read_csv("data/raw/bond_market_quotes.csv")
curve_df = bootstrap_zero_coupon_curve(quotes)
curve = ZeroCouponYieldCurve(curve_df)

five_year_zero = curve.zero_rate(5.0)
one_year_five_year_forward = curve.forward_rate(1.0, 5.0)
```

This adds a practical rates foundation for future fixed-income extensions such as curve-based bond pricing, swap discounting, DV01/PV01 analytics, and scenario-based curve shocks.

## Applying the Yield Curve to Instrument Pricing

The main risk pipeline now consumes the bootstrapped zero-coupon curve when pricing rate-sensitive instruments and option discounting.

By default, `main.py` looks for:

```text
data/raw/bond_market_quotes.csv
```

If the file exists, the pipeline bootstraps the curve and writes:

```text
data/processed/zero_coupon_curve.csv
```

Then the same curve is injected into `MarketContext` and used by the pricing layer.

Run with curve-based pricing:

```bash
python main.py --lookback-days 60 --option-pricing-engine all
```

Use a custom bond quote file:

```bash
python main.py \
  --bond-quotes-path data/raw/bond_market_quotes.csv \
  --lookback-days 60 \
  --option-pricing-engine all
```

Disable the curve and fall back to a flat risk-free rate:

```bash
python main.py --no-yield-curve
```

Curve usage by instrument:

| Instrument | Curve Usage |
| --- | --- |
| `European Option` | Uses maturity-specific zero rate for Black-Scholes, Monte Carlo, binomial, and trinomial engines |
| `FX Forward` | Uses domestic discount factor from the zero curve and foreign flat curve assumption |
| `Equity Future` | Discounts forward payoff using the curve discount factor |
| `Zero Coupon Bond` | Prices directly from `face_value × discount_factor(maturity)` |
| `Fixed Rate Bond` | Discounts each coupon and principal cash flow using curve discount factors |
| `Interest Rate Swap` | Uses curve discount factors for annuity and curve-implied forward/par rate for floating leg approximation |

Core API:

```python
import pandas as pd
from src.pricing.instruments import MarketContext, price_fixed_rate_bond
from src.pricing.yield_curve import ZeroCouponYieldCurve, bootstrap_zero_coupon_curve

quotes = pd.read_csv("data/raw/bond_market_quotes.csv")
zero_curve = ZeroCouponYieldCurve(bootstrap_zero_coupon_curve(quotes))

ctx = MarketContext(
    valuation_date=pd.Timestamp("2025-04-04"),
    risk_free_rate=0.02,          # fallback only
    dividend_yield=0.00,
    vols={"AAPL": 0.20},
    fx_foreign_rates={"EURUSD=X": 0.015},
    zero_curve=zero_curve,
)

bond = price_fixed_rate_bond(
    quantity=1,
    face_value=1000,
    coupon_rate=0.045,
    maturity=pd.Timestamp("2030-04-04"),
    ctx=ctx,
    yield_rate=None,             # None means use the zero curve
    frequency=2,
)
```

This makes the pricing stack closer to a front-office rates workflow: market bond quotes bootstrap the discount curve, and the curve feeds valuation, risk sensitivities, and scenario revaluation.

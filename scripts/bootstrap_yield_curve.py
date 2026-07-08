from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.pricing.yield_curve import (
    ZeroCouponYieldCurve,
    bootstrap_zero_coupon_curve,
    sample_bond_market_quotes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct a zero-coupon yield curve from bond market data")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/bond_market_quotes.csv"),
        help="CSV file with InstrumentID, MaturityYears, CouponRate, MarketPrice, optional FaceValue/Frequency",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/zero_coupon_curve.csv"),
        help="Output CSV for bootstrapped zero-coupon curve",
    )
    parser.add_argument(
        "--write-sample-input",
        action="store_true",
        help="Write sample bond market quotes before bootstrapping",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.write_sample_input or not args.input.exists():
        sample_bond_market_quotes().to_csv(args.input, index=False)

    quotes = pd.read_csv(args.input)
    zero_curve = bootstrap_zero_coupon_curve(quotes)
    zero_curve.to_csv(args.output, index=False)

    curve = ZeroCouponYieldCurve(zero_curve)
    summary = zero_curve[["InstrumentID", "MaturityYears", "DiscountFactor", "ZeroRate"]].copy()
    summary["ZeroRatePct"] = summary["ZeroRate"] * 100.0
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.6f}"))
    print(f"\nWrote zero-coupon curve to {args.output.resolve()}")
    print(f"Example 1Y-5Y continuous forward rate: {curve.forward_rate(1.0, 5.0):.4%}")


if __name__ == "__main__":
    main()

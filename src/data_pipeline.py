"""
Data Pipeline Module
====================
Loads, validates, and cleans the raw Shopify e-commerce CSV.
Handles returned-order adjustments for accurate CLV computation.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def load_raw_data(filepath=None):
    """Load the raw CSV and parse dates."""
    filepath = filepath or config.RAW_CSV
    df = pd.read_csv(filepath, parse_dates=["order_date"])
    print(f"[Pipeline] Loaded {len(df):,} rows, {df['customer_id'].nunique():,} customers")
    return df


def validate_data(df):
    """Run validation checks and print a summary report."""
    report = {}

    # Null check
    null_counts = df.isnull().sum()
    report["total_nulls"] = int(null_counts.sum())

    # Date range
    report["date_min"] = str(df["order_date"].min().date())
    report["date_max"] = str(df["order_date"].max().date())

    # Revenue sanity: revenue ≈ discounted_price × quantity
    expected_revenue = df["discounted_price"] * df["quantity"]
    revenue_mismatch = (df["revenue"] - expected_revenue).abs() > 0.01
    report["revenue_mismatches"] = int(revenue_mismatch.sum())

    # Negative profit
    report["negative_profit_rows"] = int((df["profit"] < 0).sum())

    # Return rate
    report["return_rate"] = round(df["is_returned"].mean() * 100, 2)

    print("[Pipeline] Validation Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    return report


def clean_data(df):
    """
    Clean and prepare the dataset.

    Key decision: We do NOT drop returned orders. Instead, we flag them
    so CLV calculations can net out returns. This is more realistic than
    dropping rows — a customer who buys $500 and returns $200 has a CLV
    of $300, not $500.
    """
    df = df.copy()

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df["order_date"]):
        df["order_date"] = pd.to_datetime(df["order_date"])

    # Create net_revenue: zero out revenue for returned orders
    df["net_revenue"] = np.where(df["is_returned"] == 1, 0.0, df["revenue"])
    df["net_profit"] = np.where(df["is_returned"] == 1, 0.0, df["profit"])

    # Standardize string columns
    for col in ["product_category", "customer_country", "traffic_source", "payment_method"]:
        df[col] = df[col].str.strip().str.title()

    # Sort by customer and date for time-based features
    df = df.sort_values(["customer_id", "order_date"]).reset_index(drop=True)

    print(f"[Pipeline] Cleaned: {len(df):,} rows, added net_revenue/net_profit columns")
    return df


def run_pipeline(filepath=None):
    """Full pipeline: load → validate → clean → return."""
    df = load_raw_data(filepath)
    validate_data(df)
    df = clean_data(df)
    return df


if __name__ == "__main__":
    df = run_pipeline()
    print(f"\nFinal shape: {df.shape}")
    print(df.head())

"""
RFM Feature Engineering Module
===============================
Aggregates transaction-level data (60K rows) into customer-level
features (31K rows) for clustering and CLV prediction.

Goes beyond standard RFM with behavioral features:
- Return rate, category diversity, avg discount, channel preference,
  basket size — giving SHAP richer features to explain.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def compute_rfm_features(df, snapshot_date=None):
    """
    Aggregate transaction data to customer-level RFM+ features.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction-level data with net_revenue column.
    snapshot_date : pd.Timestamp, optional
        Reference date for recency. Defaults to max(order_date) + 1 day.

    Returns
    -------
    pd.DataFrame
        Customer-level feature table (one row per customer_id).
    """
    if snapshot_date is None:
        snapshot_date = df["order_date"].max() + pd.Timedelta(days=1)

    print(f"[Features] Snapshot date: {snapshot_date.date()}")
    print(f"[Features] Aggregating {len(df):,} transactions → customer features...")

    # ── Core RFM ────────────────────────────────────────────────
    agg = df.groupby("customer_id").agg(
        recency=("order_date", lambda x: (snapshot_date - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("net_revenue", "mean"),  # avg order value (net of returns)
        total_clv=("net_revenue", "sum"),  # target variable
        total_revenue_gross=("revenue", "sum"),
        first_order=("order_date", "min"),
        last_order=("order_date", "max"),
    ).reset_index()

    # ── Tenure ──────────────────────────────────────────────────
    agg["tenure_days"] = (agg["last_order"] - agg["first_order"]).dt.days
    # For single-order customers, tenure = 0; add 1 to avoid division issues
    agg["tenure_days_safe"] = agg["tenure_days"].clip(lower=1)

    # ── Behavioral Features ─────────────────────────────────────
    behavioral = df.groupby("customer_id").agg(
        avg_discount=("discount_percent", "mean"),
        avg_quantity=("quantity", "mean"),
        avg_rating=("rating", "mean"),
        avg_shipping=("shipping_cost", "mean"),
        return_rate=("is_returned", "mean"),
        total_items=("quantity", "sum"),
        n_returns=("is_returned", "sum"),
        category_diversity=("product_category", "nunique"),
        avg_product_price=("product_price", "mean"),
    ).reset_index()

    agg = agg.merge(behavioral, on="customer_id", how="left")

    # ── Dominant Category (mode) ────────────────────────────────
    cat_mode = (
        df.groupby("customer_id")["product_category"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
        .rename(columns={"product_category": "dominant_category"})
    )
    agg = agg.merge(cat_mode, on="customer_id", how="left")

    # ── Dominant Traffic Source ──────────────────────────────────
    source_mode = (
        df.groupby("customer_id")["traffic_source"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
        .rename(columns={"traffic_source": "dominant_channel"})
    )
    agg = agg.merge(source_mode, on="customer_id", how="left")

    # ── Dominant Country ────────────────────────────────────────
    country_mode = (
        df.groupby("customer_id")["customer_country"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
        .rename(columns={"customer_country": "country"})
    )
    agg = agg.merge(country_mode, on="customer_id", how="left")

    # ── Dominant Payment Method ─────────────────────────────────
    pay_mode = (
        df.groupby("customer_id")["payment_method"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
        .rename(columns={"payment_method": "dominant_payment"})
    )
    agg = agg.merge(pay_mode, on="customer_id", how="left")

    # ── Derived Ratios ──────────────────────────────────────────
    agg["purchase_rate"] = agg["frequency"] / agg["tenure_days_safe"]  # orders per day
    agg["avg_order_value"] = agg["total_clv"] / agg["frequency"].clip(lower=1)

    # Drop helper columns
    agg = agg.drop(columns=["first_order", "last_order", "tenure_days_safe"])

    # Round floats for cleanliness
    float_cols = agg.select_dtypes(include="float64").columns
    agg[float_cols] = agg[float_cols].round(4)

    print(f"[Features] Generated {len(agg):,} customer profiles with {len(agg.columns)} features")
    return agg


def save_rfm_features(rfm_df, path=None):
    """Save RFM features to processed folder."""
    path = path or config.RFM_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    rfm_df.to_csv(path, index=False)
    print(f"[Features] Saved to {path}")


if __name__ == "__main__":
    from data_pipeline import run_pipeline

    df = run_pipeline()
    rfm = compute_rfm_features(df)
    save_rfm_features(rfm)
    print(f"\nRFM shape: {rfm.shape}")
    print(rfm.head())
    print(rfm.describe())

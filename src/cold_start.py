"""
Cold-Start CLV Module
======================
The hardest and most differentiating component.

Solves: "How do you predict CLV for a customer with only 1-2 transactions?"

Approach: Hybrid ensemble blending probabilistic models (BG/NBD + Gamma-Gamma
from the `lifetimes` library) with the ML regressor (XGBoost).

Blend weight α varies by transaction count:
  - 1 order  → α=0.7  (lean on probabilistic priors)
  - 2 orders → α=0.5  (equal blend)
  - 3+ orders → α=0.3 (trust ML model more)
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

warnings.filterwarnings("ignore", category=FutureWarning)


def prepare_lifetimes_data(df):
    """
    Prepare data in the format required by the lifetimes library.

    lifetimes expects a summary table per customer:
    - frequency: number of REPEAT purchases (total orders - 1)
    - recency: time between first and last purchase (in days)
    - T: time between first purchase and end of observation (in days)
    - monetary_value: average order value (of repeat purchases only)
    """
    from lifetimes.utils import summary_data_from_transaction_data

    # Use only non-returned transactions for probabilistic models
    tx_data = df[df["is_returned"] == 0].copy()

    summary = summary_data_from_transaction_data(
        tx_data,
        customer_id_col="customer_id",
        datetime_col="order_date",
        monetary_value_col="revenue",
    )

    # Filter: lifetimes requires frequency > 0 for Gamma-Gamma fitting,
    # but we keep all customers for prediction
    print(f"[Cold-Start] Prepared lifetimes summary: {len(summary):,} customers")
    print(f"  Customers with repeat purchases: {(summary['frequency'] > 0).sum():,}")
    print(f"  Single-purchase customers: {(summary['frequency'] == 0).sum():,}")

    return summary


def train_bgnbd(summary):
    """
    Train BG/NBD model for predicting future transaction frequency.

    The BG/NBD (Beta-Geometric/Negative Binomial Distribution) model
    estimates:
    - P(alive): probability that a customer is still active
    - E[future_purchases]: expected number of purchases in a given period
    """
    from lifetimes import BetaGeoFitter

    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])

    print(f"[Cold-Start] BG/NBD trained successfully")
    print(f"  Model params: r={bgf.params_['r']:.4f}, alpha={bgf.params_['alpha']:.2f}, a={bgf.params_['a']:.4f}, b={bgf.params_['b']:.4f}")

    return bgf


def train_gamma_gamma(summary):
    """
    Train Gamma-Gamma model for predicting average transaction value.

    The Gamma-Gamma model estimates E[monetary_value | alive] —
    the expected average transaction value for each customer,
    accounting for individual-level heterogeneity.

    Requires: frequency > 0 (need repeat transactions to estimate)
    """
    from lifetimes import GammaGammaFitter

    # Filter to customers with repeat purchases for fitting
    returning = summary[summary["frequency"] > 0].copy()

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(returning["frequency"], returning["monetary_value"])

    print(f"[Cold-Start] Gamma-Gamma trained on {len(returning):,} repeat customers")

    return ggf


def predict_probabilistic_clv(bgf, ggf, summary, months=12):
    """
    Predict CLV using the probabilistic pipeline (BG/NBD + Gamma-Gamma).

    Parameters
    ----------
    bgf : BetaGeoFitter
        Trained BG/NBD model.
    ggf : GammaGammaFitter
        Trained Gamma-Gamma model.
    summary : pd.DataFrame
        Lifetimes summary data.
    months : int
        Prediction horizon in months.

    Returns
    -------
    pd.Series: predicted CLV per customer (indexed by customer_id)
    """
    days = months * 30  # approximate

    # Predict expected purchases in next `days` period
    summary = summary.copy()
    summary["predicted_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        days,
        summary["frequency"],
        summary["recency"],
        summary["T"],
    )

    # P(alive) for each customer
    summary["p_alive"] = bgf.conditional_probability_alive(
        summary["frequency"],
        summary["recency"],
        summary["T"],
    )

    # For customers with repeat purchases, use Gamma-Gamma for CLV
    returning_mask = summary["frequency"] > 0
    summary["prob_clv"] = 0.0

    if returning_mask.any():
        returning_data = summary[returning_mask]
        clv_values = ggf.customer_lifetime_value(
            bgf,
            returning_data["frequency"],
            returning_data["recency"],
            returning_data["T"],
            returning_data["monetary_value"],
            time=months,
            discount_rate=0.01,
        )
        summary.loc[returning_mask, "prob_clv"] = clv_values.values

    # For single-purchase customers: use population-level estimates
    single_mask = summary["frequency"] == 0
    if single_mask.any():
        avg_monetary = summary.loc[returning_mask, "monetary_value"].median()
        summary.loc[single_mask, "prob_clv"] = (
            summary.loc[single_mask, "predicted_purchases"] * avg_monetary * summary.loc[single_mask, "p_alive"]
        )

    summary["prob_clv"] = summary["prob_clv"].clip(lower=0)

    print(f"[Cold-Start] Probabilistic CLV predicted for {len(summary):,} customers")
    print(f"  Mean prob CLV: ${summary['prob_clv'].mean():,.2f}")

    return summary[["predicted_purchases", "p_alive", "prob_clv"]]


def blend_predictions(prob_clv_series, ml_clv_series, frequency_series):
    """
    Blend probabilistic and ML predictions using confidence-weighted ensemble.

    α = blend weight for probabilistic model:
      - frequency 0-1 → α=0.7 (lean on probabilistic priors)
      - frequency 2   → α=0.5 (equal blend)
      - frequency 3+  → α=0.3 (trust ML more)

    Parameters
    ----------
    prob_clv_series : pd.Series indexed by customer_id
    ml_clv_series : pd.Series indexed by customer_id
    frequency_series : pd.Series indexed by customer_id (original order count)

    Returns
    -------
    pd.DataFrame with blended_clv, confidence_score, blend_alpha
    """
    # Align indices
    common = prob_clv_series.index.intersection(ml_clv_series.index)
    prob = prob_clv_series.loc[common]
    ml = ml_clv_series.loc[common]
    freq = frequency_series.loc[common]

    # Compute alpha based on frequency
    alpha = pd.Series(0.5, index=common)
    alpha[freq <= 1] = 0.7
    alpha[freq == 2] = 0.5
    alpha[freq >= 3] = 0.3

    # Blend
    blended = alpha * prob + (1 - alpha) * ml
    blended = blended.clip(lower=0)

    # Confidence score: higher frequency + agreement between models → higher confidence
    prediction_agreement = 1 - (np.abs(prob - ml) / (np.abs(prob) + np.abs(ml) + 1))
    confidence = (np.log1p(freq) / np.log1p(freq.max())) * 0.6 + prediction_agreement * 0.4
    confidence = confidence.clip(0, 1)

    result = pd.DataFrame({
        "prob_clv": prob.round(2),
        "ml_clv": ml.round(2),
        "blended_clv": blended.round(2),
        "blend_alpha": alpha.round(2),
        "confidence_score": confidence.round(3),
    }, index=common)

    print(f"[Cold-Start] Blended CLV for {len(result):,} customers")
    print(f"  Mean blended CLV: ${result['blended_clv'].mean():,.2f}")
    print(f"  Mean confidence: {result['confidence_score'].mean():.3f}")

    return result


def save_cold_start_models(bgf, ggf):
    """Save BG/NBD and Gamma-Gamma models to disk."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bgf.save_model(config.BGNBD_PATH)
    ggf.save_model(config.GAMMA_GAMMA_PATH)
    print(f"[Cold-Start] Saved BG/NBD and Gamma-Gamma models")

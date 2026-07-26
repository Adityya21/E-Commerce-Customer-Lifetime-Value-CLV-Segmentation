"""
ROI Calculator Module
======================
Estimates revenue impact of retention campaigns targeted at
specific customer segments.

Core question: "If we improve retention by X% in segment Y,
how much additional revenue do we preserve over Z months?"
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def calculate_roi(segment_profile, retention_uplift_pct, months=12, churn_rate_base=0.25):
    """
    Calculate the estimated revenue impact of a retention campaign.

    Parameters
    ----------
    segment_profile : dict
        Segment profile with size, avg_clv, etc.
    retention_uplift_pct : float
        Expected retention improvement (e.g., 10 for 10%).
    months : int
        Projection period.
    churn_rate_base : float
        Baseline annual churn rate (default 25%).

    Returns
    -------
    dict with revenue projections and impact estimates.
    """
    segment_size = segment_profile["size"]
    avg_clv = segment_profile["avg_clv"]
    total_segment_revenue = segment_profile["total_revenue"]

    # Annualize churn rate, then scale to projection period
    monthly_churn = 1 - (1 - churn_rate_base) ** (1 / 12)
    period_churn = 1 - (1 - monthly_churn) ** months

    # Expected churners without intervention
    expected_churners_base = int(segment_size * period_churn)

    # Revenue at risk (without intervention)
    revenue_at_risk = expected_churners_base * avg_clv

    # With retention campaign
    retention_uplift = retention_uplift_pct / 100
    new_churn_rate = period_churn * (1 - retention_uplift)
    expected_churners_after = int(segment_size * new_churn_rate)

    # Customers saved
    customers_saved = expected_churners_base - expected_churners_after

    # Revenue preserved
    revenue_preserved = customers_saved * avg_clv

    # Assuming campaign cost is ~10% of revenue preserved (conservative)
    estimated_campaign_cost = revenue_preserved * 0.10
    net_roi = revenue_preserved - estimated_campaign_cost
    roi_multiplier = revenue_preserved / max(estimated_campaign_cost, 1)

    result = {
        "segment_name": segment_profile["persona_name"],
        "segment_size": segment_size,
        "avg_clv": round(avg_clv, 2),
        "projection_months": months,
        "base_churn_rate": round(period_churn * 100, 1),
        "expected_churners_without_action": expected_churners_base,
        "revenue_at_risk": round(revenue_at_risk, 2),
        "retention_uplift_pct": retention_uplift_pct,
        "new_churn_rate": round(new_churn_rate * 100, 1),
        "expected_churners_with_campaign": expected_churners_after,
        "customers_saved": customers_saved,
        "revenue_preserved": round(revenue_preserved, 2),
        "estimated_campaign_cost": round(estimated_campaign_cost, 2),
        "net_roi": round(net_roi, 2),
        "roi_multiplier": round(roi_multiplier, 1),
    }

    return result


def get_all_segments_roi(profiles, retention_uplift_pct=15, months=12):
    """
    Calculate ROI for all segments at a given retention uplift.

    Returns
    -------
    list of ROI dicts, sorted by revenue_at_risk descending.
    """
    results = []
    for seg_id, profile in profiles.items():
        roi = calculate_roi(profile, retention_uplift_pct, months)
        roi["segment_id"] = int(seg_id)
        results.append(roi)

    results.sort(key=lambda x: x["revenue_at_risk"], reverse=True)
    return results

"""
SHAP Explainability Module
============================
Provides global, per-segment, and per-customer SHAP analysis
on the CLV regression model.

SHAP (SHapley Additive exPlanations) answers:
"Which features drive this customer's predicted CLV up or down?"
"""

import pandas as pd
import numpy as np
import joblib
import shap
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def compute_shap_values(model, X_scaled, feature_names, max_samples=2000):
    """
    Compute SHAP values for the CLV model.

    Uses TreeExplainer for tree-based models (XGBoost/RF),
    falls back to KernelExplainer for others.

    Parameters
    ----------
    model : trained sklearn/xgb model
    X_scaled : np.ndarray, scaled feature matrix
    feature_names : list of str
    max_samples : int, subsample for speed

    Returns
    -------
    shap.Explanation object
    """
    if max_samples and len(X_scaled) > max_samples:
        idx = np.random.RandomState(42).choice(len(X_scaled), max_samples, replace=False)
        X_sample = X_scaled[idx]
    else:
        X_sample = X_scaled

    try:
        # TreeExplainer is fast and exact for tree models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        print(f"[SHAP] Used TreeExplainer on {len(X_sample):,} samples")
    except Exception:
        # Fallback for non-tree models
        explainer = shap.KernelExplainer(model.predict, X_sample[:100])
        shap_values = explainer.shap_values(X_sample)
        print(f"[SHAP] Used KernelExplainer on {len(X_sample):,} samples")

    explanation = shap.Explanation(
        values=shap_values,
        data=X_sample,
        feature_names=feature_names,
    )

    return explanation, explainer


def get_global_importance(shap_explanation, feature_names, top_n=10):
    """
    Get global feature importance rankings from SHAP values.

    Returns
    -------
    list of dicts: [{feature, importance, direction}, ...]
    """
    mean_abs = np.abs(shap_explanation.values).mean(axis=0)
    mean_signed = shap_explanation.values.mean(axis=0)

    importance = []
    for i, feat in enumerate(feature_names):
        importance.append({
            "feature": feat,
            "importance": round(float(mean_abs[i]), 4),
            "direction": "positive" if mean_signed[i] > 0 else "negative",
        })

    importance.sort(key=lambda x: x["importance"], reverse=True)

    print(f"\n[SHAP] Global Feature Importance (Top {top_n}):")
    for item in importance[:top_n]:
        arrow = "↑" if item["direction"] == "positive" else "↓"
        print(f"  {arrow} {item['feature']}: {item['importance']:.4f}")

    return importance[:top_n]


def get_customer_explanation(model, scaler, customer_features, feature_names):
    """
    Generate a per-customer SHAP waterfall explanation.

    Parameters
    ----------
    customer_features : dict
        Feature values for one customer.

    Returns
    -------
    dict: {predicted_clv, base_value, contributions: [{feature, value, shap_value}, ...]}
    """
    X = pd.DataFrame([customer_features])[feature_names]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_scaled = scaler.transform(X)

    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_scaled)[0]
        base = float(explainer.expected_value)
    except Exception:
        return {"error": "SHAP explanation unavailable for this model type"}

    pred = float(model.predict(X_scaled)[0])

    contributions = []
    for i, feat in enumerate(feature_names):
        contributions.append({
            "feature": feat,
            "feature_value": round(float(X_scaled[0][i]), 4),
            "shap_value": round(float(sv[i]), 2),
        })

    # Sort by absolute SHAP value
    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return {
        "predicted_clv": round(pred, 2),
        "base_value": round(base, 2),
        "contributions": contributions,
    }


def get_segment_shap(shap_explanation, labels, feature_names, profiles):
    """
    Compute average SHAP values per segment to explain
    what drives CLV for each customer group.

    Returns
    -------
    dict: segment_id → list of {feature, avg_shap_value}
    """
    segment_shap = {}

    for seg_id in sorted(set(labels)):
        mask = np.array(labels) == seg_id
        if not mask.any():
            continue

        seg_shap = shap_explanation.values[mask].mean(axis=0)
        features = []
        for i, feat in enumerate(feature_names):
            features.append({
                "feature": feat,
                "avg_shap_value": round(float(seg_shap[i]), 4),
            })
        features.sort(key=lambda x: abs(x["avg_shap_value"]), reverse=True)

        persona = profiles.get(int(seg_id), {}).get("persona_name", f"Segment {seg_id}")
        segment_shap[int(seg_id)] = {
            "persona": persona,
            "top_drivers": features[:5],
        }

    return segment_shap


def save_shap_artifacts(shap_explanation, global_importance, segment_shap):
    """Save SHAP data for the Flask app."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    shap_data = {
        "global_importance": global_importance,
        "segment_shap": segment_shap,
    }

    with open(config.PROCESSED_DIR / "shap_analysis.json", "w") as f:
        json.dump(shap_data, f, indent=2)

    # Save explanation object for potential plot generation
    joblib.dump(shap_explanation, config.SHAP_VALUES_PATH)

    print(f"[SHAP] Saved analysis and explanation object")

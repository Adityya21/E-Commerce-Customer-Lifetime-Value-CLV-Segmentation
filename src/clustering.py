"""
Customer Segmentation Module
==============================
K-Means clustering on scaled RFM features with automatic
optimal-k selection via Elbow + Silhouette analysis.
Generates business-friendly segment profiles and persona mappings.
"""

import pandas as pd
import numpy as np
import json
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# Clustering features (core RFM + key behavioral)
CLUSTER_FEATURES = ["recency", "frequency", "monetary", "tenure_days", "return_rate", "category_diversity"]


def find_optimal_k(rfm_df, k_range=range(2, 11)):
    """
    Run Elbow + Silhouette analysis to find optimal number of clusters.

    Returns
    -------
    dict with 'inertias', 'silhouettes', 'optimal_k'
    """
    X = rfm_df[CLUSTER_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        silhouettes.append(sil)
        print(f"  k={k}: Inertia={km.inertia_:,.0f}, Silhouette={sil:.4f}")

    # Pick k with highest silhouette (simple but effective)
    best_idx = np.argmax(silhouettes)
    optimal_k = list(k_range)[best_idx]

    print(f"\n[Clustering] Optimal k = {optimal_k} (Silhouette = {silhouettes[best_idx]:.4f})")

    return {
        "k_range": list(k_range),
        "inertias": inertias,
        "silhouettes": silhouettes,
        "optimal_k": optimal_k,
        "best_silhouette": silhouettes[best_idx],
    }


def train_kmeans(rfm_df, n_clusters=None, k_analysis=None):
    """
    Train K-Means with the optimal k. Returns model, scaler, labels.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        Customer-level RFM features.
    n_clusters : int, optional
        Force a specific k. If None, uses k_analysis result or defaults to 5.
    k_analysis : dict, optional
        Output from find_optimal_k().

    Returns
    -------
    tuple: (kmeans_model, scaler, labels, n_clusters)
    """
    if n_clusters is None:
        n_clusters = k_analysis["optimal_k"] if k_analysis else 5

    X = rfm_df[CLUSTER_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    print(f"[Clustering] Trained K-Means: k={n_clusters}, Silhouette={sil:.4f}")

    return kmeans, scaler, labels, n_clusters


def profile_segments(rfm_df, labels):
    """
    Generate descriptive profiles for each segment.

    Returns
    -------
    dict: segment_id → profile dict
    """
    df = rfm_df.copy()
    df["segment"] = labels

    profiles = {}
    for seg_id in sorted(df["segment"].unique()):
        seg_data = df[df["segment"] == seg_id]
        persona = config.SEGMENT_PERSONAS.get(seg_id, {"name": f"Segment {seg_id}", "emoji": "📊", "color": "#888"})

        profile = {
            "segment_id": int(seg_id),
            "persona_name": persona["name"],
            "emoji": persona["emoji"],
            "color": persona["color"],
            "size": int(len(seg_data)),
            "pct_of_base": round(len(seg_data) / len(df) * 100, 1),
            "avg_recency": round(seg_data["recency"].mean(), 1),
            "avg_frequency": round(seg_data["frequency"].mean(), 2),
            "avg_monetary": round(seg_data["monetary"].mean(), 2),
            "avg_clv": round(seg_data["total_clv"].mean(), 2),
            "total_revenue": round(seg_data["total_clv"].sum(), 2),
            "avg_tenure_days": round(seg_data["tenure_days"].mean(), 1),
            "avg_return_rate": round(seg_data["return_rate"].mean() * 100, 1),
            "avg_category_diversity": round(seg_data["category_diversity"].mean(), 2),
            "avg_discount": round(seg_data["avg_discount"].mean(), 1),
            "avg_rating": round(seg_data["avg_rating"].mean(), 2),
            "top_category": seg_data["dominant_category"].mode().iloc[0] if "dominant_category" in seg_data.columns else "N/A",
            "top_channel": seg_data["dominant_channel"].mode().iloc[0] if "dominant_channel" in seg_data.columns else "N/A",
        }
        profiles[int(seg_id)] = profile

    return profiles


def assign_persona_names(profiles):
    """
    Reorder segment personas based on actual cluster characteristics.
    Sorts segments by avg_clv and avg_frequency to assign meaningful names.
    """
    # Sort segments by CLV descending to map to personas logically
    sorted_segs = sorted(profiles.values(), key=lambda x: (-x["avg_clv"], -x["avg_frequency"]))

    persona_order = [
        {"name": "Premium Loyalists", "emoji": "🏆", "color": "#ffd700"},
        {"name": "Rising Champions", "emoji": "🌱", "color": "#66bb6a"},
        {"name": "Deal Seekers", "emoji": "🏷️", "color": "#4fc3f7"},
        {"name": "Dormant Browsers", "emoji": "💤", "color": "#78909c"},
        {"name": "At-Risk Whales", "emoji": "⚠️", "color": "#ff6b6b"},
    ]

    # Identify the at-risk segment (high monetary but high recency = gone quiet)
    # Move the segment with highest recency + decent monetary to "At-Risk Whales"
    high_recency_segs = sorted(sorted_segs, key=lambda x: -x["avg_recency"])
    at_risk_candidate = None
    for seg in high_recency_segs:
        if seg["avg_monetary"] > np.median([s["avg_monetary"] for s in sorted_segs]):
            at_risk_candidate = seg["segment_id"]
            break

    updated_profiles = {}
    persona_idx = 0
    for seg_info in sorted_segs:
        seg_id = seg_info["segment_id"]
        if seg_id == at_risk_candidate:
            persona = {"name": "At-Risk Whales", "emoji": "⚠️", "color": "#ff6b6b"}
        else:
            if persona_idx < len(persona_order):
                persona = persona_order[persona_idx]
                # Skip "At-Risk Whales" in the normal order
                while persona["name"] == "At-Risk Whales" and persona_idx < len(persona_order) - 1:
                    persona_idx += 1
                    persona = persona_order[persona_idx]
            else:
                persona = {"name": f"Segment {seg_id}", "emoji": "📊", "color": "#888888"}
            persona_idx += 1

        profiles[seg_id]["persona_name"] = persona["name"]
        profiles[seg_id]["emoji"] = persona["emoji"]
        profiles[seg_id]["color"] = persona["color"]
        updated_profiles[seg_id] = profiles[seg_id]

    return updated_profiles


def save_clustering_artifacts(kmeans, scaler, profiles, labels, rfm_df):
    """Save all clustering artifacts to disk."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(kmeans, config.KMEANS_PATH)
    joblib.dump(scaler, config.CLUSTER_SCALER_PATH)

    with open(config.SEGMENT_PROFILES_PATH, "w") as f:
        json.dump(profiles, f, indent=2)

    # Save customer → segment mapping
    seg_df = rfm_df[["customer_id"]].copy()
    seg_df["segment"] = labels
    for seg_id, prof in profiles.items():
        seg_df.loc[seg_df["segment"] == seg_id, "persona_name"] = prof["persona_name"]
    seg_df.to_csv(config.SEGMENTS_CSV, index=False)

    print(f"[Clustering] Saved: model, scaler, profiles, segments CSV")


def predict_segment(customer_features, kmeans_model=None, scaler=None):
    """Predict segment for a new customer given their RFM features."""
    if kmeans_model is None:
        kmeans_model = joblib.load(config.KMEANS_PATH)
    if scaler is None:
        scaler = joblib.load(config.CLUSTER_SCALER_PATH)

    X = pd.DataFrame([customer_features])[CLUSTER_FEATURES]
    X_scaled = scaler.transform(X)
    label = kmeans_model.predict(X_scaled)[0]
    return int(label)

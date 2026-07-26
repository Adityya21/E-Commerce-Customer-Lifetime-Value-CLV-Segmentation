"""
Training Pipeline
==================
End-to-end script that runs all ML steps and saves artifacts.
Run this once to generate all models and processed data for the Flask app.

Usage:
    python train_pipeline.py
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from src.data_pipeline import run_pipeline
from src.feature_engineering import compute_rfm_features, save_rfm_features
from src.clustering import (
    find_optimal_k, train_kmeans, profile_segments,
    assign_persona_names, save_clustering_artifacts,
)
from src.clv_model import (
    prepare_features, train_and_compare, interpret_metrics,
    save_model_artifacts,
)
from src.cold_start import (
    prepare_lifetimes_data, train_bgnbd, train_gamma_gamma,
    predict_probabilistic_clv, blend_predictions, save_cold_start_models,
)
from src.explainability import (
    compute_shap_values, get_global_importance,
    get_segment_shap, save_shap_artifacts,
)


def main():
    print("=" * 70)
    print("  CLV & SEGMENTATION - TRAINING PIPELINE")
    print("=" * 70)

    # Ensure output dirs exist
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # PHASE 1: DATA PIPELINE
    # ------------------------------------------------------------
    print("\n" + "-" * 50)
    print("PHASE 1: Data Pipeline")
    print("-" * 50)
    df = run_pipeline()

    # ------------------------------------------------------------
    # PHASE 2: RFM FEATURE ENGINEERING
    # ------------------------------------------------------------
    print("\n" + "-" * 50)
    print("PHASE 2: RFM Feature Engineering")
    print("-" * 50)
    rfm = compute_rfm_features(df)
    save_rfm_features(rfm)

    # ------------------------------------------------------------
    # PHASE 3: CUSTOMER SEGMENTATION
    # ------------------------------------------------------------
    print("\n" + "-" * 50)
    print("PHASE 3: Customer Segmentation")
    print("-" * 50)
    k_analysis = find_optimal_k(rfm)
    kmeans, cluster_scaler, labels, n_clusters = train_kmeans(rfm, k_analysis=k_analysis)
    profiles = profile_segments(rfm, labels)
    profiles = assign_persona_names(profiles)
    save_clustering_artifacts(kmeans, cluster_scaler, profiles, labels, rfm)

    print("\nSegment Profiles:")
    for seg_id, prof in profiles.items():
        print(f"  {prof['emoji']} {prof['persona_name']}: "
              f"{prof['size']:,} customers, "
              f"Avg CLV=${prof['avg_clv']:,.0f}, "
              f"Recency={prof['avg_recency']:.0f}d")

    # ------------------------------------------------------------
    # PHASE 4: CLV REGRESSION
    # ------------------------------------------------------------
    print("\n" + "-" * 50)
    print("PHASE 4: CLV Regression Model")
    print("-" * 50)
    X, y, feature_cols, label_encoders = prepare_features(rfm)
    result = train_and_compare(X, y)
    save_model_artifacts(result, label_encoders, feature_cols)

    # Business interpretations
    interpretations = interpret_metrics(result["metrics"])
    print("\nBusiness Interpretations:")
    for metric, interp in interpretations.items():
        print(f"  {metric.upper()}: {interp}")

    # Save metrics
    metrics_data = {
        "best_model": result["best_name"],
        "metrics": result["metrics"],
        "interpretations": interpretations,
        "k_analysis": {
            "optimal_k": k_analysis["optimal_k"],
            "best_silhouette": round(k_analysis["best_silhouette"], 4),
            "silhouettes": [round(s, 4) for s in k_analysis["silhouettes"]],
            "inertias": [round(i, 2) for i in k_analysis["inertias"]],
            "k_range": k_analysis["k_range"],
        },
    }
    with open(config.PROCESSED_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)

    # ------------------------------------------------------------
    # PHASE 5: COLD-START CLV
    # ------------------------------------------------------------
    print("\n" + "-" * 50)
    print("PHASE 5: Cold-Start CLV (BG/NBD + Gamma-Gamma)")
    print("-" * 50)
    lifetimes_summary = prepare_lifetimes_data(df)
    bgf = train_bgnbd(lifetimes_summary)
    ggf = train_gamma_gamma(lifetimes_summary)
    save_cold_start_models(bgf, ggf)

    # Predict probabilistic CLV
    prob_results = predict_probabilistic_clv(bgf, ggf, lifetimes_summary, months=12)

    # Get ML predictions for blending
    X_all, y_all, _, _ = prepare_features(rfm)
    X_all_scaled = result["scaler"].transform(X_all)
    ml_preds = result["best_model"].predict(X_all_scaled)
    ml_preds = np.clip(ml_preds, 0, None)
    ml_clv_series = pd.Series(ml_preds, index=rfm["customer_id"].values)

    # Blend
    prob_clv_series = prob_results["prob_clv"]
    freq_series = rfm.set_index("customer_id")["frequency"]

    blended = blend_predictions(prob_clv_series, ml_clv_series, freq_series)

    # Save CLV predictions
    clv_output = rfm[["customer_id", "frequency", "total_clv"]].copy()
    clv_output = clv_output.set_index("customer_id")
    clv_output = clv_output.join(blended[["blended_clv", "confidence_score", "blend_alpha"]], how="left")
    clv_output["ml_predicted_clv"] = ml_clv_series
    clv_output = clv_output.fillna({"blended_clv": 0, "confidence_score": 0, "blend_alpha": 0.5})
    clv_output.to_csv(config.CLV_PREDICTIONS_CSV)
    print(f"\n[Pipeline] Saved CLV predictions for {len(clv_output):,} customers")

    # Cold-start specific stats
    cold_start_mask = clv_output["frequency"] <= 1
    print(f"\nCold-Start Stats:")
    print(f"  Single-order customers: {cold_start_mask.sum():,}")
    print(f"  Avg blended CLV (cold-start): ${clv_output.loc[cold_start_mask, 'blended_clv'].mean():,.2f}")
    print(f"  Avg confidence (cold-start): {clv_output.loc[cold_start_mask, 'confidence_score'].mean():.3f}")

    # ------------------------------------------------------------
    # PHASE 6: SHAP EXPLAINABILITY
    # ------------------------------------------------------------
    print("\n" + "-" * 50)
    print("PHASE 6: SHAP Explainability")
    print("-" * 50)
    shap_explanation, _ = compute_shap_values(
        result["best_model"], result["X_test_scaled"], feature_cols, max_samples=2000
    )
    global_importance = get_global_importance(shap_explanation, feature_cols)

    # Per-segment SHAP — use integer positional indices from test set
    test_iloc = list(range(min(2000, len(result["X_test"]))))
    n_shap = len(shap_explanation.values)
    # labels array is aligned with rfm rows; map test positional indices safely
    all_labels_arr = np.array(labels)
    test_pos = result["X_test"].index.tolist()[:n_shap]
    sampled_labels = all_labels_arr[test_pos] if max(test_pos) < len(all_labels_arr) else all_labels_arr[:n_shap]
    segment_shap = get_segment_shap(shap_explanation, sampled_labels, feature_cols, profiles)
    save_shap_artifacts(shap_explanation, global_importance, segment_shap)

    # ------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"\n  Dataset: {len(df):,} transactions -> {len(rfm):,} customers")
    print(f"  Segments: {n_clusters} clusters (Silhouette = {k_analysis['best_silhouette']:.4f})")
    print(f"  Best CLV Model: {result['best_name']}")
    print(f"    R2:   {result['metrics'][result['best_name']]['r2']}")
    print(f"    RMSE: ${result['metrics'][result['best_name']]['rmse']:,}")
    print(f"    MAPE: {result['metrics'][result['best_name']]['mape']}%")
    print(f"  Cold-Start: BG/NBD + Gamma-Gamma trained")
    print(f"  SHAP: {len(global_importance)} features analyzed")
    print(f"\n  All artifacts saved to: {config.MODELS_DIR}")
    print(f"  Processed data saved to: {config.PROCESSED_DIR}")
    print(f"\n  [OK] Ready to launch Flask app: python app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()

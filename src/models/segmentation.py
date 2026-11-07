"""International customer segmentation (notebook cells 236-249)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE

CLUSTER_FEATURES = ["total_gross_amount", "purchase_frequency", "sku_diversity",
                    "style_diversity", "active_months"]
FINAL_K = 2


def customer_features(intl: pd.DataFrame) -> pd.DataFrame:
    """Cell 237."""
    out = (
        intl.groupby("customer")
        .agg(
            total_pieces=("pcs", "sum"),
            total_gross_amount=("gross_amt", "sum"),
            purchase_frequency=("customer", "size"),
            sku_diversity=("sku_standardized", "nunique"),
            style_diversity=("style", "nunique"),
            active_months=("months", "nunique"),
            sku_missing_records=("sku_standardized", lambda x: x.isna().sum()),
        )
        .reset_index()
    )
    out["avg_gross_per_record"] = out["total_gross_amount"] / out["purchase_frequency"]
    out["avg_pieces_per_record"] = out["total_pieces"] / out["purchase_frequency"]
    return out


def _kmeans(k: int) -> KMeans:
    return KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)


def run_segmentation(intl: pd.DataFrame) -> dict[str, Any]:
    feats = customer_features(intl)
    X_log = np.log1p(feats[CLUSTER_FEATURES])                             # cell 242
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X_log), columns=CLUSTER_FEATURES, index=feats.index)

    sweep = []                                                            # cell 243
    for k in range(2, 11):
        km = _kmeans(k)
        labels = km.fit_predict(X)
        sweep.append({"k": k, "inertia": km.inertia_,
                      "silhouette": silhouette_score(X, labels)})
    sweep = pd.DataFrame(sweep)

    candidates = {}                                                       # cells 244-245
    for k in (2, 6):
        km = _kmeans(k)
        candidates[k] = (km, km.fit_predict(X))
    model, labels = candidates[FINAL_K]

    seg = feats.copy()                                                    # cell 246
    seg["cluster"] = labels
    profile = (
        seg.groupby("cluster")
        .agg(
            customer_count=("customer", "count"),
            total_pieces_mean=("total_pieces", "mean"),
            total_gross_mean=("total_gross_amount", "mean"),
            total_gross_median=("total_gross_amount", "median"),
            purchase_frequency_mean=("purchase_frequency", "mean"),
            sku_diversity_mean=("sku_diversity", "mean"),
            style_diversity_mean=("style_diversity", "mean"),
            active_months_mean=("active_months", "mean"),
            gross_amount_sum=("total_gross_amount", "sum"),
        )
    )
    profile["customer_pct"] = profile["customer_count"] / len(seg) * 100
    profile["gross_share_pct"] = profile["gross_amount_sum"] / profile["gross_amount_sum"].sum() * 100

    # Names come from the profile, not the cluster id (cell 249 warns to verify order).
    high = profile["total_gross_mean"].idxmax()
    names = {c: ("High-value, high-activity customers" if c == high
                 else "Lower-value, occasional customers") for c in profile.index}
    seg["segment"] = seg["cluster"].map(names)
    profile["segment"] = profile.index.map(names)

    heat_cols = ["total_pieces_mean", "total_gross_mean", "purchase_frequency_mean",
                 "sku_diversity_mean", "style_diversity_mean", "active_months_mean"]
    hp = profile[heat_cols]
    heatmap = ((hp - hp.mean()) / hp.std()).T                             # cell 247

    k6_sizes = pd.Series(candidates[6][1]).value_counts().sort_index()
    return {
        "customers": seg, "profile": profile.reset_index(), "sweep": sweep,
        "heatmap": heatmap, "k6_sizes": k6_sizes.rename("customers").rename_axis("cluster").reset_index(),
        "silhouette": float(silhouette_score(X, labels)),
        "cluster_names": {int(k): v for k, v in names.items()},
        "high_value_cluster": int(high),
        "model": model, "scaler": scaler,
    }

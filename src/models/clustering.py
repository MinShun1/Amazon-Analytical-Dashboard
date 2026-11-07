"""Product price segmentation with K-Means (notebook cells 301-305, 309-324).

Note: despite sitting under "Product" in the notebook, this clusters the
*May-2022 pricing* SKUs (complete marketplace prices only), not the
inventory table. Only the preparation steps of the excluded Price-Analysis
section are reused here.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import MARKETPLACE_COLS, RANDOM_STATE

FINAL_K = 5
RAW_FEATURES = ["final_mrp", "tp_to_mrp_ratio", "marketplace_price_cv"]
MODEL_FEATURES = ["final_mrp_log", "tp_to_mrp_ratio", "marketplace_cv_log"]

# Names assigned in cell 323, keyed by the characteristic that defines them.
LABELS = {
    "premium": "Premium Price Products",
    "non_standard": "Non-Standard Pricing Products",
    "high_tp": "High TP-to-MRP Ratio Products",
    "variation": "Marketplace Price-Variation Products",
    "standard": "Standard Mid-Price Products",
}
NOTEBOOK_IDS = {0: "standard", 1: "high_tp", 2: "variation", 3: "non_standard", 4: "premium"}


def pricing_cluster_dataset(may_2022: pd.DataFrame) -> pd.DataFrame:
    """Cells 301, 305, 309: May-2022 SKUs with all 8 marketplace prices."""
    pricing = may_2022.copy()
    complete = pricing[["sku_standardized", "category"] + MARKETPLACE_COLS].dropna(
        subset=MARKETPLACE_COLS).copy()
    lookup = pricing.set_index("sku_standardized")
    df = complete[["sku_standardized", "category"]].copy()
    df["tp"] = lookup.loc[df["sku_standardized"], "tp"].values
    df["final_mrp"] = lookup.loc[df["sku_standardized"], "final_mrp_old"].values
    prices = complete[MARKETPLACE_COLS]
    df["avg_marketplace_price"] = prices.mean(axis=1).values
    df["median_marketplace_price"] = prices.median(axis=1).values
    df["marketplace_price_std"] = prices.std(axis=1).values
    df["marketplace_price_range"] = (prices.max(axis=1) - prices.min(axis=1)).values
    df["marketplace_price_cv"] = df["marketplace_price_std"] / df["avg_marketplace_price"] * 100
    df["tp_to_mrp_ratio"] = df["tp"] / df["final_mrp"]
    df["marketplace_to_mrp_ratio"] = df["avg_marketplace_price"] / df["final_mrp"]
    return df.reset_index(drop=True)


def _kmeans(k: int) -> KMeans:
    return KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)


def _name_clusters(profile: pd.DataFrame) -> dict[int, str]:
    """Assign the notebook's names from each cluster's profile, so the names
    stay correct even if K-Means numbers the clusters differently."""
    remaining = set(profile.index)
    keys: dict[int, str] = {}
    for key, col, pick in [("premium", "avg_final_mrp", "max"),
                           ("non_standard", "product_count", "min"),
                           ("high_tp", "avg_tp_ratio", "max"),
                           ("variation", "avg_marketplace_cv", "max")]:
        sub = profile.loc[sorted(remaining), col]
        cid = int(sub.idxmax() if pick == "max" else sub.idxmin())
        keys[cid] = key
        remaining.discard(cid)
    for cid in remaining:
        keys[int(cid)] = "standard"
    return keys


def run_pricing_clustering(may_2022: pd.DataFrame) -> dict[str, Any]:
    df = pricing_cluster_dataset(may_2022)
    X = pd.DataFrame({                                                    # cell 313
        "final_mrp_log": np.log1p(df["final_mrp"]),
        "tp_to_mrp_ratio": df["tp_to_mrp_ratio"],
        "marketplace_cv_log": np.log1p(df["marketplace_price_cv"]),
    })
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    sweep = []                                                            # cell 314
    for k in range(2, 11):
        km = _kmeans(k)
        labels = km.fit_predict(Xs)
        sweep.append({"k": k, "inertia": km.inertia_,
                      "silhouette": silhouette_score(Xs, labels),
                      "davies_bouldin": davies_bouldin_score(Xs, labels)})
    sweep = pd.DataFrame(sweep)

    candidate_sizes = []                                                  # cell 315
    for k in (5, 6, 10):
        counts = pd.Series(_kmeans(k).fit_predict(Xs)).value_counts().sort_index()
        for cid, n in counts.items():
            candidate_sizes.append({"k": k, "cluster": int(cid), "products": int(n),
                                    "pct": n / len(Xs) * 100})

    model = _kmeans(FINAL_K)                                              # cell 316
    df["cluster"] = model.fit_predict(Xs)
    profile = df.groupby("cluster").agg(
        product_count=("sku_standardized", "count"),
        avg_final_mrp=("final_mrp", "mean"),
        median_final_mrp=("final_mrp", "median"),
        avg_tp=("tp", "mean"),
        median_tp=("tp", "median"),
        avg_tp_ratio=("tp_to_mrp_ratio", "mean"),
        avg_marketplace_cv=("marketplace_price_cv", "mean"),
        median_marketplace_cv=("marketplace_price_cv", "median"),
    )
    profile["pct"] = profile["product_count"] / len(df) * 100

    keys = _name_clusters(profile)
    names = {cid: LABELS[key] for cid, key in keys.items()}
    df["cluster_label"] = df["cluster"].map(names)
    profile["cluster_label"] = profile.index.map(names)
    ids_match_notebook = all(NOTEBOOK_IDS.get(cid) == key for cid, key in keys.items())

    category_pct = pd.crosstab(df["cluster"], df["category"], normalize="index") * 100  # cell 317

    pca = PCA(n_components=2)                                             # cell 322
    coords = pca.fit_transform(Xs)
    df["PC1"], df["PC2"] = coords[:, 0], coords[:, 1]

    return {
        "products": df, "profile": profile.reset_index(), "sweep": sweep,
        "candidate_sizes": pd.DataFrame(candidate_sizes), "category_pct": category_pct,
        "pca_explained": pca.explained_variance_ratio_.tolist(),
        "silhouette": float(silhouette_score(Xs, df["cluster"])),
        "davies_bouldin": float(davies_bouldin_score(Xs, df["cluster"])),
        "cluster_names": {int(k): v for k, v in names.items()},
        "ids_match_notebook": bool(ids_match_notebook),
        "model": model, "scaler": scaler, "pca": pca,
    }

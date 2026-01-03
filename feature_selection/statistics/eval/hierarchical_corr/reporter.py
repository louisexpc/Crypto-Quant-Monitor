#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, required=True, help="results dir (feature_selection/results/hierarchical_corr)")
    ap.add_argument("--prefix", type=str, required=True, help="prefix used by run_hcorr.py")
    args = ap.parse_args()

    base_dir = Path(args.dir)
    prefix = args.prefix
    out_dir = base_dir / prefix
    if not out_dir.exists():
        out_dir = base_dir

    dist_csv = out_dir / f"{prefix}_dist.csv"
    clus_csv = out_dir / f"{prefix}_clusters.csv"
    corr_csv = out_dir / f"{prefix}_corr.csv"

    D = pd.read_csv(dist_csv, index_col=0)
    clusters = pd.read_csv(clus_csv)
    C = pd.read_csv(corr_csv, index_col=0) if corr_csv.exists() else None

    # 指標（以 precomputed distance 算 silhouette）
    labels = clusters["cluster"].values
    # 需把對角線清 0，並確保距離非負
    D_mat = D.values.copy()
    np.fill_diagonal(D_mat, 0.0)
    sil = silhouette_score(D_mat, labels, metric="precomputed")

    metrics = {"silhouette_precomputed": float(sil)}
    (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("[OK] metrics:", metrics)

    # 附贈：群大小長條圖
    counts = clusters["cluster"].value_counts().sort_index()
    cluster_ids = counts.index.astype(int)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(cluster_ids, counts.values, width=0.8, align="center")
    if len(cluster_ids) > 0:
        tick_start = (cluster_ids.min() // 5) * 5
        tick_end = cluster_ids.max()
        ticks = list(range(tick_start, tick_end + 1, 5))
        if tick_end not in ticks:
            ticks.append(tick_end)
        ax.set_xticks(ticks)
    ax.set_xlabel("cluster id")
    ax.set_ylabel("#features")
    ax.set_title("Cluster sizes")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_cluster_sizes.png", dpi=160)
    plt.close(fig)

if __name__ == "__main__":
    main()

"""
python feature_selection/statistics/hierarchical_corr/reporter.py --dir feature_selection/results/hierarchical_corr \
    --prefix hcorr_pearson_avg_k60
"""

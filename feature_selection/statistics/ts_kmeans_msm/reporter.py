from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import silhouette_score

from feature_selection.statistics.ts_kmeans_msm.ts_kmeans_msm import (
    _NUMBA_AVAILABLE,
    _msm_distance_nb,
    _progress,
    msm_distance,
)

if _NUMBA_AVAILABLE:
    try:
        from numba import njit, prange
    except Exception:
        njit = None
        prange = None
        _NUMBA_AVAILABLE = False
else:
    njit = None
    prange = None


if _NUMBA_AVAILABLE and _msm_distance_nb is not None and njit is not None and prange is not None:

    @njit(parallel=True, fastmath=True, cache=True)
    def _pairwise_msm_distance_nb(X: np.ndarray, cost: float) -> np.ndarray:
        n = X.shape[0]
        D = np.zeros((n, n), dtype=np.float32)
        for i in prange(n):
            for j in range(i + 1, n):
                d = _msm_distance_nb(X[i], X[j], cost)
                D[i, j] = d
                D[j, i] = d
        return D

else:
    _pairwise_msm_distance_nb = None  # type: ignore


def _load_cfg(path: str) -> Dict:
    """
    1. 說明: 讀取 YAML 設定檔。
    2. inputs:
       - path: 設定檔路徑
    3. return:
       - 解析後的設定字典
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_clean_matrix(run_dir: Path, prefix: str) -> Tuple[np.ndarray, List[str]]:
    """
    1. 說明: 從已清理的矩陣 CSV 讀取資料與樣本 id。
    2. inputs:
       - run_dir: 本次實驗的輸出資料夾
       - prefix: 檔名前綴
    3. return:
       - X: numpy 矩陣
       - ids: 樣本 id 串列
    """
    path = run_dir / f"{prefix}_clean_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到清理後矩陣檔案: {path}")
    df = pd.read_csv(path)
    ids = df["id"].astype(str).tolist() if "id" in df.columns else [str(i) for i in range(len(df))]
    time_cols = [c for c in df.columns if c != "id"]
    X = df[time_cols].to_numpy(dtype=float)
    return X, ids


def _plot_cluster_sizes(counts: pd.Series, out_png: Path) -> None:
    """
    1. 說明: 畫出各群大小長條圖。
    2. inputs:
       - counts: 群大小的 Series
       - out_png: 圖片輸出路徑
    3. return:
       - 無
    """
    cluster_ids = counts.index.to_numpy()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(cluster_ids, counts.values, width=0.8, align="center")
    if len(cluster_ids) > 0:
        tick_start = (cluster_ids.min() // 5) * 5
        tick_end = cluster_ids.max()
        ticks = list(range(tick_start, tick_end + 1, 5))
        if tick_end not in ticks:
            ticks.append(int(tick_end))
        ax.set_xticks(ticks)
    ax.set_xlabel("cluster id")
    ax.set_ylabel("#series")
    ax.set_title("Cluster sizes (MSM k-means)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_centroids(centroids: pd.DataFrame, out_png: Path) -> None:
    """
    1. 說明: 以折線圖呈現群中心。
    2. inputs:
       - centroids: 含 cluster_id 與時間步欄位的 DataFrame
       - out_png: 圖片輸出路徑
    3. return:
       - 無
    """
    time_cols = [c for c in centroids.columns if c != "cluster_id"]
    t_axis = np.arange(len(time_cols))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for _, row in centroids.iterrows():
        ax.plot(t_axis, row[time_cols].to_numpy(dtype=float), label=f"cluster {int(row['cluster_id'])}")
    # Clip y-axis to主要分佈，避免少數離群值讓尺度過度擴張
    vals = centroids[time_cols].to_numpy(dtype=float).ravel()
    if vals.size:
        lo, hi = np.percentile(vals, [1, 99])
        span = hi - lo
        pad = max(span * 0.1, 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("time step")
    ax.set_ylabel("value")
    ax.set_title("Cluster centroids (MSM k-means)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _silhouette_msm(X: np.ndarray, labels: np.ndarray, msm_cost: float, use_tqdm: bool) -> float:
    """
    1. 說明: 以 MSM 距離計算 silhouette 分數。
    2. inputs:
       - X: 形狀 (n_samples, seq_len) 的資料矩陣
       - labels: 群標籤
       - msm_cost: MSM 固定成本 c
       - use_tqdm: 是否啟用 tqdm 進度條
    3. return:
       - silhouette 分數
    """
    n = X.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in _progress(range(n), use_tqdm, "silhouette (rows)"):
        for j in range(i + 1, n):
            d = msm_distance(X[i], X[j], msm_cost)
            D[i, j] = D[j, i] = d
    np.fill_diagonal(D, 0.0)
    return float(silhouette_score(D, labels, metric="precomputed"))


def main() -> None:
    """
    1. 說明: 讀取 k-means MSM 結果並產生報表與指標。
    2. inputs:
       - 由 argparse 取得的設定與輸出路徑
    3. return:
       - 無
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", type=str, default="feature_selection/statistics/ts_kmeans_msm/config.yaml")
    ap.add_argument("--dir", type=str, help="覆寫輸出資料夾 (預設使用 config.yaml)")
    ap.add_argument("--prefix", type=str, help="覆寫輸出前綴 (預設使用 config.yaml)")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    out_dir = Path(args.dir) if args.dir else Path(cfg["output"]["dir"])
    prefix = args.prefix or cfg["output"]["prefix"]
    run_dir = out_dir / prefix

    labels_df = pd.read_csv(run_dir / f"{prefix}_labels.csv")
    centroids_df = pd.read_csv(run_dir / f"{prefix}_centroids.csv")
    counts = labels_df["cluster"].value_counts().sort_index()

    metrics = {}
    cluster_stats = None
    if cfg["report"].get("compute_silhouette", True):
        rep_cfg = cfg.get("report", {})
        base_limit = rep_cfg.get("silhouette_sample_limit")
        per_cluster = rep_cfg.get("silhouette_limit_per_cluster")
        n_clusters = cfg.get("cluster", {}).get("n_clusters")
        dyn_limit = None
        if per_cluster is not None and n_clusters is not None:
            try:
                dyn_limit = int(per_cluster) * int(n_clusters)
            except Exception:
                dyn_limit = None
        limits = [int(base_limit)] if base_limit is not None else []
        if dyn_limit is not None:
            limits.append(int(dyn_limit))
        sample_limit = max(limits) if limits else 400
        labels_arr = labels_df["cluster"].to_numpy()
        uniq = np.unique(labels_arr)
        if len(uniq) < 2 or len(uniq) >= len(labels_arr):
            print(f"[WARN] 群數 {len(uniq)} 不符合 silhouette 需求 (需 2 到 n_samples-1)，跳過計算。")
        elif len(labels_df) > sample_limit:
            print(f"[WARN] 樣本數 {len(labels_df)} 超過 silhouette 計算上限 {sample_limit}，跳過計算。")
        else:
            X, _ids = _load_clean_matrix(run_dir, prefix)
            n = X.shape[0]
            if _pairwise_msm_distance_nb is not None:
                # numba 並行版，直接生成對稱距離矩陣（無 tqdm）
                D = _pairwise_msm_distance_nb(X.astype(np.float32), float(cfg["cluster"].get("msm_cost", 0.1)))
            else:
                D = np.zeros((n, n), dtype=float)
                for i in _progress(range(n), cfg["cluster"].get("use_tqdm", False), "silhouette (rows)"):
                    for j in range(i + 1, n):
                        d = msm_distance(X[i], X[j], cfg["cluster"].get("msm_cost", 0.1))
                        D[i, j] = D[j, i] = d
                np.fill_diagonal(D, 0.0)
            sil = float(silhouette_score(D, labels_arr, metric="precomputed"))
            metrics["silhouette_msm"] = sil
            # cluster-wise intra/inter mean and silhouette
            cluster_stats = []
            for cid in uniq:
                idx = np.where(labels_arr == cid)[0]
                size = int(len(idx))
                intra = None
                if size > 1:
                    sub = D[np.ix_(idx, idx)]
                    tri = sub[np.triu_indices_from(sub, k=1)]
                    intra = float(np.mean(tri)) if tri.size else None
                inter_mean = None
                other_cids = [c for c in uniq if c != cid]
                if other_cids:
                    d_means = []
                    for oc in other_cids:
                        oidx = np.where(labels_arr == oc)[0]
                        block = D[np.ix_(idx, oidx)]
                        if block.size > 0:
                            d_means.append(float(np.mean(block)))
                    if d_means:
                        inter_mean = float(min(d_means))
                sil_c = None
                if intra is not None and inter_mean is not None and max(intra, inter_mean) > 0:
                    sil_c = float((inter_mean - intra) / max(intra, inter_mean))
                cluster_stats.append(
                    {
                        "cluster_id": int(cid),
                        "size": size,
                        "intra_mean": intra,
                        "inter_mean": inter_mean,
                        "silhouette": sil_c,
                    }
                )
            metrics["cluster_stats"] = cluster_stats
            if cluster_stats:
                intra_vals = [c["intra_mean"] for c in cluster_stats if c["intra_mean"] is not None]
                inter_vals = [c["inter_mean"] for c in cluster_stats if c["inter_mean"] is not None]
                sil_vals = [c["silhouette"] for c in cluster_stats if c["silhouette"] is not None]
                if intra_vals:
                    metrics["cluster_intra_mean_avg"] = float(np.mean(intra_vals))
                if inter_vals:
                    metrics["cluster_inter_mean_avg"] = float(np.mean(inter_vals))
                if sil_vals:
                    metrics["cluster_silhouette_mean"] = float(np.mean(sil_vals))

    if metrics:
        (run_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("[OK] metrics:", metrics)

    if cfg["report"].get("make_cluster_sizes_plot", True):
        _plot_cluster_sizes(counts, run_dir / f"{prefix}_cluster_sizes.png")
    if cfg["report"].get("make_centroid_plot", True):
        _plot_centroids(centroids_df, run_dir / f"{prefix}_centroids.png")


if __name__ == "__main__":
    main()

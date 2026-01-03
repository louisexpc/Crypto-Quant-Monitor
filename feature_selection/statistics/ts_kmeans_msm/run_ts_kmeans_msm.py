from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from feature_selection.statistics.ts_kmeans_msm.ts_kmeans_msm import TimeSeriesKMeansMSM, msm_distance


def _to_utc(ts: str | None):
    """
    1. 說明: 將時間字串轉為 UTC 時區的 Timestamp。
    2. inputs:
       - ts: 時間字串或 None
    3. return:
       - UTC Timestamp 或 None
    """
    if not ts:
        return None
    parsed = pd.Timestamp(ts)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed


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


def _prepare_matrix(cfg: Dict) -> Tuple[np.ndarray, List[str], List[str], pd.DataFrame, pd.DataFrame]:
    """
    1. 說明: 依設定讀取 CSV 並轉成分群用的矩陣（每個 feature 為一筆樣本）。
    2. inputs:
       - cfg: 設定字典
    3. return:
       - X: (n_features, seq_len) 的 numpy 矩陣
       - ids: 特徵名稱列表
       - time_cols: 代表時間步的索引名稱列表
       - data: 清理/篩選/降採樣後的 DataFrame（index 為時間，columns 為特徵）
       - data_full: 清理/篩選但未降採樣/截斷的 DataFrame（保留完整時間軸）
    """
    ip = cfg["input"]
    src = Path(ip["csv_path"])
    if not src.exists():
        raise FileNotFoundError(f"[ts_kmeans_msm] 找不到輸入檔: {src}")

    df = pd.read_csv(src)
    idxcol = ip.get("index_col")
    if idxcol and idxcol in df.columns:
        df = df.set_index(idxcol)
    elif "datetime" in df.columns:
        df = df.set_index("datetime")
    elif "timestamp" in df.columns:
        df = df.set_index("timestamp")
    else:
        raise ValueError("[ts_kmeans_msm] 需要 datetime/timestamp/index_col 作為索引")

    df.index = pd.to_datetime(df.index, utc=True)
    start_ts = _to_utc(ip.get("start"))
    end_ts = _to_utc(ip.get("end"))
    if start_ts is not None:
        df = df[df.index >= start_ts]
    if end_ts is not None:
        df = df[df.index <= end_ts]
    if df.empty:
        raise ValueError("[ts_kmeans_msm] 所選時間範圍沒有資料")

    drop_na = bool(ip.get("drop_na", True))
    exclude = set(ip.get("exclude_cols") or [])
    num_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    if not num_cols:
        raise ValueError("[ts_kmeans_msm] 找不到可用的數值欄位")

    data = df[num_cols].replace([np.inf, -np.inf], np.nan)
    if drop_na:
        data = data.dropna()
    if data.isna().any().any():
        raise ValueError("[ts_kmeans_msm] 清理後仍含 NaN/Inf，請先處理資料")
    if data.empty:
        raise ValueError("[ts_kmeans_msm] 清理後無資料可用")

    data_full = data.copy()  # 保留未降採樣/截斷的版本供後續輸出完整時間軸

    downsample_every = int(ip.get("downsample_every", 1))
    if downsample_every > 1:
        data = data.iloc[::downsample_every]
        print(f"[INFO] downsample time axis every {downsample_every} rows → {len(data)} steps")

    max_ts = ip.get("max_time_steps")
    if max_ts is not None:
        max_ts = int(max_ts)
        if len(data) > max_ts:
            before = len(data)
        data = data.iloc[-max_ts:]
        print(f"[INFO] keep last {max_ts} time steps (from {before} rows)")

    # 轉置為 (n_features, seq_len)，使每個特徵成為一筆樣本
    X = data.T.to_numpy(dtype=np.float32)
    ids = list(data.columns)
    time_cols = [str(ts) for ts in data.index]
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("[ts_kmeans_msm] X 需要形狀 (n_features, seq_len) 且不可為空")
    print(f"[INFO] matrix ready: samples={X.shape[0]} seq_len={X.shape[1]}")

    return X, ids, time_cols, data, data_full


def _plot_cluster_sizes(counts: pd.Series, out_png: Path) -> None:
    """
    1. 說明: 畫出各群大小的長條圖。
    2. inputs:
       - counts: 每個群的樣本數 Series
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


def _plot_centroids(centroids: np.ndarray, out_png: Path) -> None:
    """
    1. 說明: 將群中心序列畫成折線圖。
    2. inputs:
       - centroids: (k, seq_len) 的群中心矩陣
       - out_png: 圖片輸出路徑
    3. return:
       - 無
    """
    k, seq_len = centroids.shape
    t_axis = np.arange(seq_len)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i in range(k):
        ax.plot(t_axis, centroids[i], label=f"cluster {i}")
    # Clip y-axis to central range so極端離群值不會把尺度撐爆
    flat = centroids.ravel()
    if flat.size:
        lo, hi = np.percentile(flat, [1, 99])
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


def _choose_representatives(
    ids: List[str],
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    msm_cost: float,
    use_numba: bool,
) -> pd.DataFrame:
    """選出各群離 centroid 最近的特徵（MSM 距離的 medoid）。"""
    reps = []
    labels_arr = np.asarray(labels)
    uniq = np.unique(labels_arr)
    for cid in sorted(uniq):
        idx = np.where(labels_arr == cid)[0]
        if idx.size == 0:
            continue
        if int(cid) >= centroids.shape[0]:
            raise ValueError(f"[ts_kmeans_msm] centroid 缺失，找不到群 {cid}")
        centroid = centroids[int(cid)]
        best_i = None
        best_d = None
        for i in idx:
            d = msm_distance(X[i], centroid, msm_cost, use_numba=use_numba)
            if best_d is None or d < best_d:
                best_d = d
                best_i = int(i)
        reps.append(
            {
                "cluster_id": int(cid),
                "representative": ids[int(best_i)],
                "distance_to_centroid": float(best_d),
                "size": int(len(idx)),
            }
        )
    return pd.DataFrame(reps).sort_values("cluster_id").reset_index(drop=True)


def _export_selected_features(cfg: Dict, reps: List[str], data_full: pd.DataFrame, run_dir: Path, prefix: str) -> None:
    """將代表特徵欄位擷取成方便後續使用的 CSV 與欄位清單（保留完整時間軸）。"""
    if not reps:
        print("[WARN] 代表特徵列表為空，略過 selected_feat 匯出")
        return

    df = data_full.copy()
    df_reset = df.reset_index()
    idx_name = df.index.name
    base_cols: List[str] = []
    for col in [idx_name, "datetime", "timestamp"]:
        if col and col in df_reset.columns and col not in base_cols:
            base_cols.append(col)

    keep_cols = [c for c in reps if c in df_reset.columns]
    missing = sorted(set(reps) - set(keep_cols))
    if missing:
        print(f"[WARN] {len(missing)} representatives not found in source columns; skipped.")
    if not keep_cols:
        print("[WARN] 沒有可匯出的代表特徵，略過 selected_feat 匯出")
        return

    out_df = df_reset[base_cols + keep_cols] if base_cols else df_reset[keep_cols]
    out_csv = run_dir / f"{prefix}_selected_feat.csv"
    out_df.to_csv(out_csv, index=False)

    out_txt = run_dir / f"{prefix}_selected_feat_cols.txt"
    out_txt.write_text("\n".join(keep_cols), encoding="utf-8")


def main() -> None:
    """
    1. 說明: 依 YAML 設定執行 MSM k-means，並輸出結果與圖表。
    2. inputs:
       - 由 argparse 取得的設定檔路徑
    3. return:
       - 無
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", type=str, default="feature_selection/statistics/ts_kmeans_msm/config.yaml")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    out_dir = Path(cfg["output"]["dir"])
    prefix = cfg["output"]["prefix"]
    run_dir = out_dir / prefix
    run_dir.mkdir(parents=True, exist_ok=True)

    X, ids, time_cols, data, data_full = _prepare_matrix(cfg)

    model = TimeSeriesKMeansMSM(
        n_clusters=cfg["cluster"]["n_clusters"],
        msm_cost=cfg["cluster"].get("msm_cost", 0.1),
        max_iter=cfg["cluster"].get("max_iter", 100),
        n_init=cfg["cluster"].get("n_init", 3),
        tol=cfg["cluster"].get("tol", 1e-3),
        random_state=cfg["cluster"].get("random_state"),
        use_tqdm=cfg["cluster"].get("use_tqdm", False),
        use_numba=cfg["cluster"].get("use_numba", True),
    )
    labels = model.fit_predict(X)
    centroids = model.cluster_centers_
    inertia = model.inertia_

    labels_df = pd.DataFrame({"id": ids, "cluster": labels})
    labels_df.to_csv(run_dir / f"{prefix}_labels.csv", index=False)

    centroids_df = pd.DataFrame(centroids, columns=time_cols)
    centroids_df.insert(0, "cluster_id", np.arange(model.n_clusters))
    centroids_df.to_csv(run_dir / f"{prefix}_centroids.csv", index=False)

    clean_df = pd.DataFrame(X, columns=time_cols)
    clean_df.insert(0, "id", ids)
    clean_df.to_csv(run_dir / f"{prefix}_clean_matrix.csv", index=False)

    counts = labels_df["cluster"].value_counts().sort_index()
    counts.to_csv(run_dir / f"{prefix}_cluster_sizes.csv", index=True, header=["count"])

    reps_df = _choose_representatives(
        ids=ids,
        X=X,
        labels=labels,
        centroids=centroids,
        msm_cost=cfg["cluster"].get("msm_cost", 0.1),
        use_numba=model._use_numba_backend,
    )
    reps_df.to_csv(run_dir / f"{prefix}_representatives.csv", index=False)
    _export_selected_features(cfg, reps_df["representative"].tolist(), data_full, run_dir, prefix)

    meta = {
        "n_samples": int(X.shape[0]),
        "seq_len": int(X.shape[1]),
        "n_clusters": int(model.n_clusters),
        "msm_cost": float(model.msm_cost),
        "max_iter": int(model.max_iter),
        "n_init": int(model.n_init),
        "tol": float(model.tol),
        "random_state": model.random_state,
        "inertia": float(inertia),
        "use_numba_requested": bool(model.use_numba),
        "use_numba_effective": bool(getattr(model, "_use_numba_backend", False)),
        "input_csv": str(cfg["input"]["csv_path"]),
        "representatives": reps_df.to_dict(orient="records"),
    }
    (run_dir / f"{prefix}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if cfg["report"].get("make_cluster_sizes_plot", True):
        _plot_cluster_sizes(counts, run_dir / f"{prefix}_cluster_sizes.png")
    if cfg["report"].get("make_centroid_plot", True):
        _plot_centroids(centroids, run_dir / f"{prefix}_centroids.png")

    print(f"[OK] results → {run_dir}")


if __name__ == "__main__":
    main()

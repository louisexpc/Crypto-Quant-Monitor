#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import math
import fnmatch

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score

try:
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.metrics import cdist_soft_dtw
    from tslearn.utils import to_time_series_dataset
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "tslearn is required for soft-DTW k-means. Install via `pip install tslearn`."
    ) from exc


def _load_cfg(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _prepare_matrix(cfg: dict) -> Tuple[pd.DataFrame, List[str], pd.Index | None]:
    ip = cfg["input"]
    df = pd.read_csv(ip["csv_path"])
    idxcol = ip.get("index_col")
    if idxcol and idxcol in df.columns:
        df = df.set_index(idxcol)

    excl = set(ip.get("exclude_cols") or [])
    num_cols = [
        c
        for c in df.columns
        if c not in excl and np.issubdtype(df[c].dtype, np.number)
    ]
    original_count = len(num_cols)
    num_cols = _filter_feature_list(num_cols, ip)
    if not num_cols:
        raise ValueError("No numeric feature columns selected after filtering.")

    X = df[num_cols].copy()
    if X.isna().any().any():
        raise ValueError("[softdtw_kmeans] 特徵矩陣含 NaN/Inf，請先處理資料後再分析。")
    if ip.get("drop_na", True):
        X = X.dropna()

    time_index = X.index if isinstance(X.index, pd.Index) else None
    print(
        f"[INFO] softdtw_kmeans: selected {len(num_cols)} features "
        f"(from {original_count} numeric columns)."
    )
    return X, num_cols, time_index


def _filter_feature_list(cols: Sequence[str], ip_cfg: dict) -> List[str]:
    include_patterns = ip_cfg.get("include_patterns") or []
    include_exact = ip_cfg.get("include_cols") or []

    if include_patterns:
        matched = []
        for pat in include_patterns:
            matched.extend(fnmatch.filter(cols, pat))
        cols = list(dict.fromkeys(matched))  # preserve order, dedupe

    if include_exact:
        allow = set(include_exact)
        cols = [c for c in cols if c in allow]

    max_features = ip_cfg.get("max_features")
    if max_features is not None:
        max_features = int(max_features)
        if max_features > 0 and len(cols) > max_features:
            strategy = str(ip_cfg.get("sample_strategy", "head")).lower()
            if strategy == "random":
                rng = np.random.default_rng(ip_cfg.get("sample_random_state", 42))
                idx = np.sort(rng.choice(len(cols), size=max_features, replace=False))
                cols = [cols[i] for i in idx]
            elif strategy == "tail":
                cols = list(cols)[-max_features:]
            else:  # head (default)
                cols = list(cols)[:max_features]
            print(
                f"[INFO] softdtw_kmeans: downsampled features to {len(cols)} "
                f"using strategy '{strategy}'."
            )
        elif max_features == 0:
            raise ValueError("max_features=0 results in empty feature set.")

    return list(cols)


def _apply_preprocess(X: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    if not cfg:
        return X
    out = X.copy()
    if cfg.get("zscore_each_series", False):
        means = out.mean(axis=0)
        stds = out.std(axis=0, ddof=0).replace(0, 1.0)
        out = (out - means) / stds
    max_steps = cfg.get("max_time_steps")
    orig_steps = len(out)
    if max_steps is not None:
        max_steps = int(max_steps)
        if max_steps > 0 and orig_steps > max_steps:
            strategy = str(cfg.get("time_sampling", "uniform")).lower()
            out = _limit_time_steps(out, max_steps, strategy)
            print(
                f"[INFO] softdtw_kmeans: reduced time steps from {orig_steps} to {len(out)} using '{strategy}'."
            )
        elif max_steps == 0:
            raise ValueError("preprocess.max_time_steps=0 would drop all rows.")
    return out


def _to_ts_dataset(X: pd.DataFrame) -> np.ndarray:
    series_list = [X[col].astype(np.float32).to_numpy() for col in X.columns]
    return to_time_series_dataset(series_list)


def _limit_time_steps(df: pd.DataFrame, max_steps: int, strategy: str) -> pd.DataFrame:
    if len(df) <= max_steps:
        return df
    if strategy == "uniform":
        idx = np.linspace(0, len(df) - 1, max_steps, dtype=int)
        return df.iloc[idx]
    if strategy == "stride":
        step = max(1, math.ceil(len(df) / max_steps))
        return df.iloc[::step].head(max_steps)
    if strategy == "head":
        return df.iloc[:max_steps]
    if strategy == "tail":
        return df.iloc[-max_steps:]
    raise ValueError(f"Unknown time_sampling strategy: {strategy}")


def _fit_softdtw_kmeans(
    ts_data: np.ndarray, cfg: dict
) -> TimeSeriesKMeans:
    km_cfg = cfg["kmeans"]
    metric_params = dict(km_cfg.get("metric_params") or {})
    if "gamma" not in metric_params:
        metric_params["gamma"] = km_cfg.get("gamma", 0.01)

    model = TimeSeriesKMeans(
        n_clusters=int(km_cfg["n_clusters"]),
        metric="softdtw",
        metric_params=metric_params,
        n_init=int(km_cfg.get("n_init", 1)),
        max_iter=int(km_cfg.get("max_iter", 50)),
        max_iter_barycenter=int(km_cfg.get("max_iter_barycenter", 20)),
        random_state=km_cfg.get("random_state"),
        n_jobs=km_cfg.get("n_jobs"),
        verbose=km_cfg.get("verbose", 0),
    )
    model.fit(ts_data)
    return model


def _compute_pairwise_softdtw(
    ts_data: np.ndarray, gamma: float
) -> np.ndarray:
    data = ts_data.squeeze(-1)
    return cdist_soft_dtw(data, data, gamma=gamma)


def _choose_representatives(
    labels: np.ndarray, distances: np.ndarray, feature_names: Sequence[str]
) -> pd.DataFrame:
    reps = []
    labels = labels.astype(int)
    unique = np.unique(labels)
    for cluster_id in unique:
        members = np.where(labels == cluster_id)[0]
        if len(members) == 0:
            continue
        cluster_dists = distances[members, cluster_id]
        local_best = int(np.argmin(cluster_dists))
        best_idx = members[local_best]
        reps.append(
            {
                "cluster_id": int(cluster_id),
                "representative": feature_names[best_idx],
                "distance": float(cluster_dists[local_best]),
                "size": int(len(members)),
            }
        )
    return pd.DataFrame(reps).sort_values("cluster_id")


def _plot_cluster_sizes(labels: np.ndarray, out_png: Path) -> None:
    counts = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel("cluster id")
    ax.set_ylabel("#features")
    ax.set_title("Cluster sizes (soft-DTW k-means)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_centroids(
    centers: np.ndarray,
    time_index: pd.Index | None,
    out_png: Path,
    max_centroids: int | None = None,
) -> None:
    k, length = centers.shape
    if max_centroids is not None:
        k = min(k, max_centroids)
    xs = np.arange(length) if time_index is None else np.asarray(time_index[:length])
    fig, ax = plt.subplots(figsize=(10, 4 + k * 0.15))
    for cid in range(k):
        ax.plot(xs, centers[cid], label=f"cluster_{cid}")
    ax.set_xlabel("time" if time_index is None else "index")
    ax.set_title("Soft-DTW k-means centroids")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _export_centroids(
    centers: np.ndarray,
    time_index: pd.Index | None,
    run_dir: Path,
    prefix: str,
) -> None:
    idx = (
        pd.Index(range(centers.shape[1]), name="step")
        if time_index is None
        else time_index
    )
    cent_df = pd.DataFrame(
        centers.T,
        index=idx,
        columns=[f"cluster_{i}" for i in range(centers.shape[0])],
    )
    cent_df.to_csv(run_dir / f"{prefix}_centroids.csv", index=True)


def _export_selected_features(
    cfg: dict, representatives: Iterable[str], run_dir: Path
) -> None:
    ip = cfg["input"]
    source_path = Path(ip["csv_path"])
    if not source_path.exists():
        print(f"[WARN] Source CSV not found for selected_feat export: {source_path}")
        return

    df = pd.read_csv(source_path)
    idxcol = ip.get("index_col")

    reps = list(representatives)
    keep_cols = [col for col in reps if col in df.columns]
    missing = sorted(set(reps) - set(keep_cols))
    if missing:
        print(f"[WARN] missing {len(missing)} representative columns in source CSV; skipped.")

    if not keep_cols:
        print("[WARN] No representative columns exported.")
        return

    base_cols: List[str] = []
    for col in [idxcol, "datetime", "timestamp"]:
        if col and col in df.columns and col not in base_cols:
            base_cols.append(col)

    selected_df = df[base_cols + keep_cols] if base_cols else df[keep_cols]
    out_csv = run_dir / f"{cfg['output']['prefix']}_selected_feat.csv"
    selected_df.to_csv(out_csv, index=False)

    out_txt = run_dir / f"{cfg['output']['prefix']}_selected_feat_cols.txt"
    out_txt.write_text("\n".join(keep_cols), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-c",
        "--config",
        type=str,
        default="softdtw_kmeans/config.yaml",
        help="Path to YAML config.",
    )
    args = ap.parse_args()
    cfg = _load_cfg(args.config)

    out_dir = Path(cfg["output"]["dir"])
    prefix = cfg["output"]["prefix"]
    run_dir = out_dir / prefix
    run_dir.mkdir(parents=True, exist_ok=True)

    X_raw, feature_names, _ = _prepare_matrix(cfg)
    X = _apply_preprocess(X_raw, cfg.get("preprocess", {}))
    time_index = X.index
    ts_data = _to_ts_dataset(X)

    model = _fit_softdtw_kmeans(ts_data, cfg)
    labels = model.labels_
    distances = model.transform(ts_data)

    metric_params = dict(cfg["kmeans"].get("metric_params") or {})
    gamma = metric_params.get("gamma", cfg["kmeans"].get("gamma", 0.01))

    report_cfg = cfg.get("report", {})

    need_pairwise = (
        report_cfg.get("compute_silhouette", True)
        or cfg["output"].get("save_distance_csv", False)
    )
    dist_matrix = None
    if need_pairwise:
        max_pairwise = int(report_cfg.get("pairwise_max_samples", 600))
        if len(feature_names) > max_pairwise:
            print(
                f"[WARN] Skipping pairwise soft-DTW distances "
                f"({len(feature_names)} features > limit {max_pairwise})."
            )
        else:
            dist_matrix = _compute_pairwise_softdtw(ts_data, gamma=gamma)

    if cfg["output"].get("save_distance_csv", False) and dist_matrix is not None:
        dist_df = pd.DataFrame(dist_matrix, index=feature_names, columns=feature_names)
        dist_df.to_csv(run_dir / f"{prefix}_dist.csv", index=True)

    reps = _choose_representatives(labels, distances, feature_names)
    reps.to_csv(run_dir / f"{prefix}_representatives.csv", index=False)

    clusters_df = pd.DataFrame({"feature": feature_names, "cluster": labels})
    clusters_df.to_csv(run_dir / f"{prefix}_clusters.csv", index=False)

    _export_selected_features(cfg, reps["representative"].tolist(), run_dir)

    centers = model.cluster_centers_.squeeze(-1)
    _export_centroids(centers, time_index, run_dir, prefix)

    if report_cfg.get("plot_cluster_sizes", True):
        _plot_cluster_sizes(labels, run_dir / f"{prefix}_cluster_sizes.png")
    if report_cfg.get("plot_centroids", True):
        _plot_centroids(
            centers,
            time_index,
            run_dir / f"{prefix}_centroids_plot.png",
            report_cfg.get("max_centroids_to_plot"),
        )

    silhouette = None
    if report_cfg.get("compute_silhouette", True) and dist_matrix is not None:
        if len(np.unique(labels)) > 1:
            silhouette = float(
                silhouette_score(dist_matrix, labels, metric="precomputed")
            )
        else:
            silhouette = float("nan")

    meta = {
        "n_features": len(feature_names),
        "n_clusters": int(cfg["kmeans"]["n_clusters"]),
        "gamma": float(gamma),
        "random_state": cfg["kmeans"].get("random_state"),
        "silhouette_precomputed": silhouette,
        "inertia": float(model.inertia_) if hasattr(model, "inertia_") else None,
    }
    (run_dir / f"{prefix}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"[OK] results → {run_dir}")


if __name__ == "__main__":
    main()

"""
python feature_selection/statistics/softdtw_kmeans/run_softdtw_kmeans.py -c feature_selection/statistics/softdtw_kmeans/config.yaml
"""

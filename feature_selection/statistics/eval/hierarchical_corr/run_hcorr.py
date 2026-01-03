# feature_selection/statistics/hierarchical_corr/run_hcorr.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform


def _load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_utc(ts: str | None):
    if not ts:
        return None
    parsed = pd.Timestamp(ts)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed


def _prepare_matrix(cfg: Dict) -> tuple[pd.DataFrame, List[str]]:
    ip = cfg["input"]
    src = Path(ip["csv_path"])
    if not src.exists():
        raise FileNotFoundError(f"[hierarchical_corr] 找不到輸入檔: {src}")

    df = pd.read_csv(src)
    idxcol = ip.get("index_col")
    if idxcol and idxcol in df.columns:
        df = df.set_index(idxcol)
    elif "datetime" in df.columns:
        df = df.set_index("datetime")
    elif "timestamp" in df.columns:
        df = df.set_index("timestamp")
    else:
        raise ValueError("[hierarchical_corr] 需要 datetime/timestamp/index_col 作為索引")

    df.index = pd.to_datetime(df.index, utc=True)

    start_ts = _to_utc(ip.get("start"))
    end_ts = _to_utc(ip.get("end"))
    if start_ts is not None:
        df = df[df.index >= start_ts]
    if end_ts is not None:
        df = df[df.index <= end_ts]
    if df.empty:
        raise ValueError("[hierarchical_corr] 所選時間範圍沒有資料")

    exclude = set(ip.get("exclude_cols") or [])
    num_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    if not num_cols:
        raise ValueError("[hierarchical_corr] 沒有可用的數值欄位")

    X = df[num_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    drop_na = bool(ip.get("drop_na", True))
    if drop_na:
        X = X.dropna()
        if X.isna().any().any():
            raise ValueError("[hierarchical_corr] dropna 後仍含 NaN/Inf，請檢查輸入資料。")
    else:
        if X.isna().any().any():
            raise ValueError("[hierarchical_corr] 特徵矩陣仍含 NaN/Inf，請先處理後再分析。")
    var = X.var(axis=0, ddof=0)
    zero_var_cols = var[var <= 0].index.tolist()
    if zero_var_cols:
        X = X.drop(columns=zero_var_cols)
        if X.empty:
            raise ValueError("[hierarchical_corr] 全部欄位皆為零變異，無法建立相關矩陣")
        print(f"[WARN] drop zero-variance columns: {len(zero_var_cols)} removed")
    if X.empty:
        raise ValueError("[hierarchical_corr] dropna 後沒有資料可分析")
    return X, list(X.columns)


def _corr_matrix(X: pd.DataFrame, method: str, clip_r: bool) -> pd.DataFrame:
    C = X.corr(method=method)
    if clip_r:
        C = C.clip(-1.0, 1.0)
    np.fill_diagonal(C.values, 1.0)
    return C


def _to_distance(C: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "mantegna":
        D = np.sqrt(2.0 * (1.0 - C.values))
    elif kind == "one_minus_abs":
        D = 1.0 - np.abs(C.values)
    else:
        raise ValueError(f"未知距離類型: {kind}")
    np.fill_diagonal(D, 0.0)
    return pd.DataFrame(D, index=C.index, columns=C.columns)


def _linkage(D: pd.DataFrame, method: str):
    condensed = squareform(D.values, checks=False)
    return linkage(condensed, method=method)


def _cut_tree(Z, n_clusters: int | None, max_d: float | None) -> np.ndarray:
    if n_clusters is not None:
        return fcluster(Z, n_clusters, criterion="maxclust")
    if max_d is not None:
        return fcluster(Z, max_d, criterion="distance")
    raise ValueError("必須設定 n_clusters 或 max_d 其中之一")


def _choose_representatives(D: pd.DataFrame, labels: np.ndarray, names: List[str], how: str) -> pd.DataFrame:
    lab = pd.Series(labels, index=names, name="cluster")
    reps = []
    for cid, members in lab.groupby(lab):
        idx = list(members.index)
        sub = D.loc[idx, idx].values
        if how == "medoid":
            mean_d = sub.mean(axis=1)
            best_idx = int(np.argmin(mean_d))
            score = float(mean_d[best_idx])
        elif how == "max_degree":
            tri = sub[np.triu_indices_from(sub, 1)]
            thr = np.quantile(tri, 0.25) if len(tri) else np.inf
            deg = (sub <= thr).sum(axis=1) - 1
            best_idx = int(np.argmax(deg))
            score = float(deg[best_idx])
        else:
            raise ValueError(f"未知代表挑選方式: {how}")
        reps.append({
            "cluster_id": int(cid),
            "representative": idx[best_idx],
            "score": score,
            "size": len(idx),
        })
    return pd.DataFrame(reps).sort_values("cluster_id")


def _plot_dendrogram(Z, names: List[str], out_png: Path):
    width = min(max(12.0, len(names) * 0.18), 48.0)
    fig, ax = plt.subplots(figsize=(width, 5.5))
    dendrogram(Z, labels=names, leaf_rotation=90, leaf_font_size=8, ax=ax)
    ax.set_title("Hierarchical clustering (correlation distance)")
    ax.set_ylabel("linkage distance")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_corr_heatmap(C: pd.DataFrame, Z, out_png: Path):
    leaves = dendrogram(Z, no_plot=True)["leaves"]
    C2 = C.iloc[leaves, :].iloc[:, leaves]
    n_feats = len(C2.columns)
    fig, ax = plt.subplots(figsize=(min(max(8.0, n_feats * 0.15), 48.0), min(max(6.0, n_feats * 0.15), 48.0)))
    im = ax.imshow(C2.values, origin="lower", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(C2.columns)))
    ax.set_yticks(range(len(C2.index)))
    ax.set_xticklabels(C2.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(C2.index, fontsize=6)
    ax.tick_params(axis="x", pad=1)
    ax.tick_params(axis="y", pad=1)
    ax.set_title("Correlation heatmap (leaf-ordered)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_repr_corr_heatmap(C: pd.DataFrame, reps: List[str], out_png: Path):
    # 代表指標間的相關性矩陣
    reps_present = [r for r in reps if r in C.index and r in C.columns]
    if len(reps_present) < 2:
        print("[WARN] 代表指標少於 2 或缺資料，略過代表相關矩陣繪圖")
        return
    sub = C.loc[reps_present, reps_present]
    n = len(sub)
    fig, ax = plt.subplots(figsize=(min(max(4.0, n * 0.4), 20.0), min(max(3.5, n * 0.4), 20.0)))
    im = ax.imshow(sub.values, origin="lower", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(reps_present, rotation=90, fontsize=7)
    ax.set_yticklabels(reps_present, fontsize=7)
    ax.set_title("Correlation heatmap (representatives)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _export_selected_features(cfg: Dict, representatives: Iterable[str], run_dir: Path) -> None:
    ip = cfg["input"]
    source_path = Path(ip["csv_path"])
    if not source_path.exists():
        print(f"[WARN] Source CSV not found for selected_feat export: {source_path}")
        return

    df = pd.read_csv(source_path)
    idxcol = ip.get("index_col")

    keep_cols = [col for col in representatives if col in df.columns]
    missing = sorted(set(representatives) - set(keep_cols))
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
    ap.add_argument("-c", "--config", type=str, default="hierarchical_corr/config.yaml")
    args = ap.parse_args()
    cfg = _load_cfg(args.config)

    out_dir = Path(cfg["output"]["dir"])
    prefix = cfg["output"]["prefix"]
    run_dir = out_dir / prefix
    run_dir.mkdir(parents=True, exist_ok=True)

    X, names = _prepare_matrix(cfg)
    C = _corr_matrix(X, cfg["corr"]["method"], cfg["corr"].get("clip_r", True))
    D = _to_distance(C, cfg["distance"]["kind"])
    Z = _linkage(D, cfg["cluster"]["linkage"])
    labels = _cut_tree(Z, cfg["cluster"].get("n_clusters"), cfg["cluster"].get("max_d"))
    reps = _choose_representatives(D, labels, names, cfg["cluster"]["representative"])

    if cfg["output"].get("save_corr_csv", True):
        C.to_csv(run_dir / f"{prefix}_corr.csv", index=True)
    if cfg["output"].get("save_dist_csv", True):
        D.to_csv(run_dir / f"{prefix}_dist.csv", index=True)

    pd.DataFrame({"feature": names, "cluster": labels}).to_csv(run_dir / f"{prefix}_clusters.csv", index=False)
    reps.to_csv(run_dir / f"{prefix}_representatives.csv", index=False)
    _export_selected_features(cfg, reps["representative"].tolist(), run_dir)

    if cfg["report"].get("make_dendrogram", True):
        _plot_dendrogram(Z, names, run_dir / f"{prefix}_dendrogram.png")
    if cfg["report"].get("make_heatmap", True):
        _plot_corr_heatmap(C, Z, run_dir / f"{prefix}_corr_heatmap.png")
    if cfg["report"].get("make_repr_corr_heatmap", True):
        _plot_repr_corr_heatmap(C, reps["representative"].tolist(), run_dir / f"{prefix}_repr_corr_heatmap.png")

    meta = {
        "n_features": len(names),
        "n_clusters": int(cfg["cluster"].get("n_clusters")) if cfg["cluster"].get("n_clusters") is not None else None,
        "linkage": cfg["cluster"]["linkage"],
        "distance": cfg["distance"]["kind"],
        "corr_method": cfg["corr"]["method"],
    }
    (run_dir / f"{prefix}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] results → {run_dir}")


if __name__ == "__main__":
    main()

"""
python feature_selection/statistics/hierarchical_corr/run_hcorr.py -c feature_selection/statistics/hierarchical_corr/config.yaml
"""

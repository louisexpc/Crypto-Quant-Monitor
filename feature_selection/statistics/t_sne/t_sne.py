# from __future__ import annotations

# import argparse
# from pathlib import Path
# from typing import List, Tuple
# import json

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.manifold import TSNE

# from feature_selection.statistics.ts_kmeans_msm.ts_kmeans_msm import _msm_distance_nb, msm_distance

# try:
#     from tqdm import tqdm
# except ImportError:
#     tqdm = None

# from sklearn.manifold import TSNE
# from feature_selection.statistics.ts_kmeans_msm.ts_kmeans_msm import _msm_distance_nb, msm_distance

# # 加入 numba 相關 import
# try:
#     from numba import jit, prange
#     HAS_NUMBA = True
# except ImportError:
#     jit = lambda x: x
#     prange = range
#     HAS_NUMBA = False


# def _load_labels(cluster_dir: Path) -> pd.DataFrame:
#     label_files = sorted(cluster_dir.glob("*_labels.csv"))
#     if not label_files:
#         raise FileNotFoundError(f"[t-sne] 找不到 *_labels.csv 於 {cluster_dir}")
#     # 如果有多個，優先選擇含資料夾名稱的
#     target_name = cluster_dir.name
#     chosen = None
#     for f in label_files:
#         if target_name in f.stem:
#             chosen = f
#             break
#     if chosen is None:
#         chosen = label_files[0]
#     labels_df = pd.read_csv(chosen)
#     if {"id", "cluster"} - set(labels_df.columns):
#         raise ValueError("[t-sne] labels 檔案需要包含 id, cluster 欄位")
#     return labels_df[["id", "cluster"]]


# def _progress(it, enable: bool, desc: str):
#     if enable and tqdm is not None:
#         return tqdm(it, desc=desc)
#     return it


# def _load_feature_matrix(ts_csv: Path) -> Tuple[np.ndarray, List[str]]:
#     df = pd.read_csv(ts_csv)
#     drop_cols = [c for c in ["datetime", "timestamp"] if c in df.columns]
#     df = df.drop(columns=drop_cols)
#     df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
#     df = df.dropna(axis=0, how="all")
#     df = df.dropna(axis=1, how="any")
#     if df.empty or df.shape[1] == 0:
#         raise ValueError("[t-sne] 時間序列矩陣為空，請檢查輸入 CSV")
#     features = list(df.columns)
#     matrix = df.to_numpy(dtype=np.float32).T  # shape = (n_features, n_time_steps)
#     return matrix, features


# def _align_with_clusters(
#     matrix: np.ndarray,
#     features: List[str],
#     labels_df: pd.DataFrame,
# ) -> Tuple[np.ndarray, List[str], np.ndarray]:
#     label_map = labels_df.set_index("id")["cluster"]
#     available = [f for f in features if f in label_map]
#     if not available:
#         raise ValueError("[t-sne] 交集為空：feature CSV 與 labels 不對齊")
#     missing = [f for f in label_map.index if f not in features]
#     if missing:
#         print(f"[WARN] {len(missing)} 個在 labels 內但 CSV 中缺少的指標已略過")
#     idx_map = {f: i for i, f in enumerate(features)}
#     sel_idx = [idx_map[f] for f in available]
#     aligned_matrix = matrix[sel_idx]
#     clusters = label_map.loc[available].to_numpy(dtype=int)
#     return aligned_matrix, available, clusters


# def _assign_clusters_all(features: List[str], labels_df: pd.DataFrame) -> np.ndarray:
#     label_map = labels_df.set_index("id")["cluster"]
#     clusters = np.full(len(features), -1, dtype=int)
#     for i, f in enumerate(features):
#         if f in label_map:
#             clusters[i] = int(label_map[f])
#     return clusters


# def _infer_corr_path(hcorr_clusters_csv: Path) -> Path:
#     stem = hcorr_clusters_csv.stem
#     if stem.endswith("_clusters"):
#         stem = stem[:-9]
#     return hcorr_clusters_csv.with_name(f"{stem}_corr.csv")


# def _mean_corr_between_reps(reps: List[str], corr_csv: Path, ts_csv: Path | None = None) -> float | None:
#     if len(reps) < 2:
#         return 0.0
#     df_corr = None
#     if corr_csv and corr_csv.exists():
#         try:
#             df_corr = pd.read_csv(corr_csv, index_col=0)
#         except Exception:
#             df_corr = None
#     if df_corr is None and ts_csv is not None:
#         raw = pd.read_csv(ts_csv)
#         cols = [c for c in reps if c in raw.columns]
#         if not cols:
#             return None
#         df_corr = raw[cols].corr()
#     if df_corr is None:
#         return None
#     present = [c for c in reps if c in df_corr.index and c in df_corr.columns]
#     if len(present) < 2:
#         return None
#     sub = df_corr.loc[present, present].to_numpy()
#     tri = sub[np.triu_indices_from(sub, k=1)]
#     if tri.size == 0:
#         return 0.0
#     return float(np.nanmean(tri))


# def _load_silhouette(cluster_dir: Path) -> float | None:
#     for f in sorted(cluster_dir.glob("*metrics.json")):
#         try:
#             data = json.loads(f.read_text())
#             if "silhouette_msm" in data:
#                 return float(data["silhouette_msm"])
#         except Exception:
#             continue
#     return None


# def _map_clusters_via_hcorr(
#     features: List[str],
#     labels_df: pd.DataFrame,
#     hcorr_clusters_csv: Path,
#     hcorr_reps_csv: Path,
# ) -> np.ndarray:
#     """
#     依據 hierarchical corr 60 群 → 代表指標 → MSM 15 群的映射，為所有 feature 指派 MSM 群。
#     """
#     hc = pd.read_csv(hcorr_clusters_csv)
#     reps = pd.read_csv(hcorr_reps_csv)
#     if {"feature", "cluster"} - set(hc.columns):
#         raise ValueError("[t-sne] hcorr_clusters_csv 需要欄位 feature, cluster")
#     if {"cluster_id", "representative"} - set(reps.columns):
#         raise ValueError("[t-sne] hcorr_reps_csv 需要欄位 cluster_id, representative")

#     feat_to_hcluster = hc.set_index("feature")["cluster"].to_dict()
#     hcluster_to_rep = reps.set_index("cluster_id")["representative"].to_dict()
#     label_map = labels_df.set_index("id")["cluster"].to_dict()

#     clusters = np.full(len(features), -1, dtype=int)
#     missing_feat = []
#     missing_rep = []
#     missing_label = []
#     for idx, f in enumerate(features):
#         hcid = feat_to_hcluster.get(f)
#         if hcid is None:
#             missing_feat.append(f)
#             continue
#         rep = hcluster_to_rep.get(hcid)
#         if rep is None:
#             missing_rep.append(f)
#             continue
#         msm_cluster = label_map.get(rep)
#         if msm_cluster is None:
#             missing_label.append(rep)
#             continue
#         clusters[idx] = int(msm_cluster)

#     if missing_feat:
#         print(f"[WARN] {len(missing_feat)} features 在 hcorr_clusters_csv 中找不到，標記為 -1")
#     if missing_rep:
#         print(f"[WARN] {len(missing_rep)} features 的 hcorr cluster 沒有代表指標，標記為 -1")
#     if missing_label:
#         uniq_missing = sorted(set(missing_label))
#         print(f"[WARN] {len(uniq_missing)} 代表指標沒有對應 MSM 群，標記為 -1: {uniq_missing[:5]}{'...' if len(uniq_missing)>5 else ''}")
#     return clusters


# def _standardize_rows(matrix: np.ndarray) -> np.ndarray:
#     mean = matrix.mean(axis=1, keepdims=True)
#     std = matrix.std(axis=1, keepdims=True) + 1e-8
#     return (matrix - mean) / std


# def _pairwise_msm_distance(matrix: np.ndarray, msm_cost: float, use_numba: bool, use_tqdm: bool) -> np.ndarray:
#     """
#     計算 pairwise MSM 距離矩陣，matrix shape = (n_features, seq_len)
#     """
#     n = matrix.shape[0]
#     dist = np.zeros((n, n), dtype=np.float32)
#     for i in _progress(range(n), enable=use_tqdm, desc="MSM pairwise"):
#         dist[i, i] = 0.0
#         for j in range(i + 1, n):
#             d = msm_distance(matrix[i], matrix[j], c=msm_cost, use_numba=use_numba)
#             dist[i, j] = dist[j, i] = d
#     return dist


# def _run_tsne(
#     data: np.ndarray,
#     metric: str,
#     perplexity: float,
#     random_state: int,
#     learning_rate: float,
#     n_iter: int,
# ) -> np.ndarray:
#     n_samples = data.shape[0]
#     max_perp = max(1.0, n_samples - 1.0)
#     perp = min(perplexity, max_perp)
#     if perp < 2.0 and n_samples > 2:
#         perp = min(5.0, max_perp)
#     tsne = TSNE(
#         n_components=2,
#         perplexity=perp,
#         random_state=random_state,
#         learning_rate=learning_rate,
#         n_iter=n_iter,
#         metric=metric,
#         init="pca",
#     )
#     return tsne.fit_transform(data)


# def _plot_embedding(
#     embedding: np.ndarray, clusters: np.ndarray, out_png: Path, title: str, label_unlabeled: bool = True
# ) -> None:
#     fig, ax = plt.subplots(figsize=(8.5, 5.5))
#     unique_clusters = sorted(np.unique(clusters).tolist())
#     cmap = plt.get_cmap("tab20")
#     color_idx = 0
#     legend_handles = []
#     for cid in unique_clusters:
#         mask = clusters == cid
#         if cid == -1 and label_unlabeled:
#             color = "#888888"
#             lbl = f"unlabeled (n={mask.sum()})"
#         else:
#             color = cmap(color_idx % cmap.N)
#             lbl = f"cluster {cid} (n={mask.sum()})"
#             color_idx += 1
#         pts = ax.scatter(
#             embedding[mask, 0],
#             embedding[mask, 1],
#             s=28,
#             color=color,
#             label=lbl,
#             alpha=0.85,
#             edgecolors="k",
#             linewidths=0.2,
#         )
#         legend_handles.append((pts, lbl))
#     ax.set_xlabel("t-SNE X")
#     ax.set_ylabel("t-SNE Y")
#     ax.set_title(title)
#     handles, labels = zip(*legend_handles) if legend_handles else ([], [])
#     ax.legend(
#         handles,
#         labels,
#         loc="center left",
#         bbox_to_anchor=(1.02, 0.5),
#         borderaxespad=0.0,
#         fontsize=8,
#         framealpha=0.9,
#     )
#     fig.tight_layout(rect=(0.0, 0.0, 0.8, 1.0))
#     fig.savefig(out_png, dpi=160, bbox_inches="tight")
#     plt.close(fig)


# def main() -> None:
#     ap = argparse.ArgumentParser()
#     ap.add_argument(
#         "--cluster_dir",
#         type=Path,
#         required=True,
#         help="MSM k-means 結果目錄 (內含 *_labels.csv)",
#     )
#     ap.add_argument(
#         "--ts_csv",
#         type=Path,
#         required=True,
#         help="時間序列特徵 CSV (列為時間步、欄為指標)",
#     )
#     ap.add_argument(
#         "--out_dir",
#         type=Path,
#         default=Path("feature_selection/results/t_sne"),
#         help="輸出目錄，預設 feature_selection/results/t_sne",
#     )
#     ap.add_argument("--perplexity", type=float, default=30.0)
#     ap.add_argument("--learning_rate", type=float, default=200.0)
#     ap.add_argument("--n_iter", type=int, default=1000)
#     ap.add_argument("--random_state", type=int, default=0)
#     ap.add_argument(
#         "--metric",
#         type=str,
#         default="euclidean",
#         choices=["euclidean", "msm"],
#         help="t-SNE 使用的距離：'euclidean' (快，原本方式) 或 'msm' (需先算 pairwise MSM)",
#     )
#     ap.add_argument("--msm_cost", type=float, default=0.1, help="MSM 距離的 c 參數")
#     ap.add_argument("--use_numba", action="store_true", help="若已安裝 numba，啟用以加速 MSM 距離")
#     ap.add_argument("--use_tqdm", action="store_true", help="顯示 tqdm 進度條")
#     ap.add_argument(
#         "--hier_clusters_csv",
#         type=Path,
#         default=None,
#         help="(選用) hierarchical corr cluster 對應檔，欄位 feature,cluster；若提供將用它 + representatives + MSM labels 為全部 feature 指派群",
#     )
#     ap.add_argument(
#         "--hier_reps_csv",
#         type=Path,
#         default=None,
#         help="(選用) hierarchical corr 代表指標檔，欄位 cluster_id,representative；需搭配 --hier_clusters_csv",
#     )
#     ap.add_argument("--alpha", type=float, default=1.0, help="score 中的 α，預設 1.0")
#     ap.add_argument("--beta", type=float, default=1.0, help="score 中的 β，預設 1.0")
#     ap.add_argument(
#         "--prefix",
#         type=str,
#         default=None,
#         help="輸出檔名前綴，預設取 cluster_dir 名稱",
#     )
#     args = ap.parse_args()

#     cluster_dir = args.cluster_dir
#     ts_csv = args.ts_csv
#     out_dir = args.out_dir
#     prefix = args.prefix or cluster_dir.name
#     run_dir = out_dir / prefix
#     run_dir.mkdir(parents=True, exist_ok=True)

#     if args.metric == "msm":
#         if args.use_tqdm and tqdm is None:
#             print("[WARN] tqdm 未安裝，將不顯示進度條")
#         if args.use_numba and _msm_distance_nb is None:
#             print("[WARN] numba 未安裝，MSM 距離將使用純 Python 版本")

#     labels_df = _load_labels(cluster_dir)
#     matrix, features = _load_feature_matrix(ts_csv)
#     matrix = _standardize_rows(matrix)

#     reps_list: List[str] | None = None
#     if args.hier_clusters_csv and args.hier_reps_csv:
#         reps_df = pd.read_csv(args.hier_reps_csv)
#         if {"cluster_id", "representative"} - set(reps_df.columns):
#             raise ValueError("[t-sne] hier_reps_csv 需包含 cluster_id, representative 欄位")
#         reps_list = reps_df["representative"].tolist()
#         clusters_full = _map_clusters_via_hcorr(features, labels_df, args.hier_clusters_csv, args.hier_reps_csv)
#     elif args.hier_clusters_csv or args.hier_reps_csv:
#         raise ValueError("[t-sne] 需同時提供 --hier_clusters_csv 與 --hier_reps_csv，或都不提供")
#     else:
#         clusters_full = _assign_clusters_all(features, labels_df)
#     feature_to_idx = {f: i for i, f in enumerate(features)}
#     reps_present: List[str] = []
#     if reps_list:
#         reps_present = [r for r in reps_list if r in feature_to_idx]
#     use_reps = len(reps_present) >= 2

#     if use_reps:
#         subset_idx = [feature_to_idx[r] for r in reps_present]
#         matrix_subset = matrix[subset_idx]
#         features_subset = reps_present
#         clusters_subset = clusters_full[subset_idx]
#         subset_title = "t-SNE of representatives (colored by MSM cluster)"
#     else:
#         mask_labeled = clusters_full != -1
#         if not mask_labeled.any():
#             raise ValueError("[t-sne] feature CSV 與 labels 完全無交集，無法上色")
#         matrix_subset = matrix[mask_labeled]
#         features_subset = [f for f, keep in zip(features, mask_labeled) if keep]
#         clusters_subset = clusters_full[mask_labeled]
#         subset_title = "t-SNE of labeled features (colored by MSM cluster)"
#         subset_idx = np.where(mask_labeled)[0]

#     if args.metric == "msm":
#         dist_full = _pairwise_msm_distance(
#             matrix,
#             msm_cost=args.msm_cost,
#             use_numba=args.use_numba,
#             use_tqdm=args.use_tqdm and tqdm is not None,
#         )
#         if use_reps:
#             subset_idx = [feature_to_idx[r] for r in reps_present]
#         else:
#             subset_idx = np.where(clusters_subset != -1)[0]  # all subset points have labels here
#         dist_subset = dist_full[np.ix_(subset_idx, subset_idx)]

#         embedding = _run_tsne(
#             dist_subset,
#             metric="precomputed",
#             perplexity=args.perplexity,
#             random_state=args.random_state,
#             learning_rate=args.learning_rate,
#             n_iter=args.n_iter,
#         )
#         embedding_full = _run_tsne(
#             dist_full,
#             metric="precomputed",
#             perplexity=args.perplexity,
#             random_state=args.random_state,
#             learning_rate=args.learning_rate,
#             n_iter=args.n_iter,
#         )
#     else:
#         embedding = _run_tsne(
#             matrix_subset,
#             metric="euclidean",
#             perplexity=args.perplexity,
#             random_state=args.random_state,
#             learning_rate=args.learning_rate,
#             n_iter=args.n_iter,
#         )
#         embedding_full = _run_tsne(
#             matrix,
#             metric="euclidean",
#             perplexity=args.perplexity,
#             random_state=args.random_state,
#             learning_rate=args.learning_rate,
#             n_iter=args.n_iter,
#         )

#     embed_df = pd.DataFrame(
#         {
#             "feature": features_subset,
#             "cluster": clusters_subset.astype(int),
#             "tsne_x": embedding[:, 0],
#             "tsne_y": embedding[:, 1],
#         }
#     )
#     embed_csv = run_dir / f"{prefix}_tsne_embedding.csv"
#     embed_df.to_csv(embed_csv, index=False)

#     stats_df = (
#         embed_df.groupby("cluster")
#         .agg(size=("feature", "count"), tsne_x_mean=("tsne_x", "mean"), tsne_y_mean=("tsne_y", "mean"))
#         .reset_index()
#         .rename(columns={"cluster": "cluster_id"})
#     )
#     stats_csv = run_dir / f"{prefix}_tsne_cluster_stats.csv"
#     stats_df.to_csv(stats_csv, index=False)

#     plot_path = run_dir / f"{prefix}_tsne.png"
#     _plot_embedding(
#         embedding,
#         clusters_subset,
#         plot_path,
#         title=subset_title,
#         label_unlabeled=not use_reps,
#     )

#     embed_full_df = pd.DataFrame(
#         {
#             "feature": features,
#             "cluster": clusters_full.astype(int),
#             "tsne_x": embedding_full[:, 0],
#             "tsne_y": embedding_full[:, 1],
#         }
#     )
#     embed_full_csv = run_dir / f"{prefix}_tsne_full_embedding.csv"
#     embed_full_df.to_csv(embed_full_csv, index=False)

#     stats_full_df = (
#         embed_full_df.groupby("cluster")
#         .agg(size=("feature", "count"), tsne_x_mean=("tsne_x", "mean"), tsne_y_mean=("tsne_y", "mean"))
#         .reset_index()
#         .rename(columns={"cluster": "cluster_id"})
#     )
#     stats_full_csv = run_dir / f"{prefix}_tsne_full_cluster_stats.csv"
#     stats_full_df.to_csv(stats_full_csv, index=False)

#     plot_full_path = run_dir / f"{prefix}_tsne_full.png"
#     _plot_embedding(
#         embedding_full,
#         clusters_full,
#         plot_full_path,
#         title="t-SNE of all features (labeled/unlabeled)",
#         label_unlabeled=True,
#     )

#     corr_path = _infer_corr_path(args.hier_clusters_csv) if args.hier_clusters_csv else None
#     mean_corr = _mean_corr_between_reps(reps_list or [], corr_path, ts_csv if ts_csv.exists() else None)
#     silhouette = _load_silhouette(cluster_dir)
#     d_final = len(reps_list) if reps_list is not None else int((clusters_full != -1).sum())
#     score = None
#     if silhouette is not None and mean_corr is not None and d_final > 0:
#         score = float(silhouette - args.alpha * mean_corr - args.beta * (d_final / 276.0))
#     score_path = run_dir / f"{prefix}_tsne_score.json"
#     score_payload = {
#         "silhouette_msm": silhouette,
#         "mean_corr_representatives": mean_corr,
#         "d_final": d_final,
#         "alpha": args.alpha,
#         "beta": args.beta,
#         "score": score,
#     }
#     score_path.write_text(json.dumps(score_payload, indent=2), encoding="utf-8")

#     print(f"[OK] t-SNE embedding (labeled) → {embed_csv}")
#     print(f"[OK] cluster stats (labeled)   → {stats_csv}")
#     print(f"[OK] plot (labeled)            → {plot_path}")
#     print(f"[OK] t-SNE embedding (full)    → {embed_full_csv}")
#     print(f"[OK] cluster stats (full)      → {stats_full_csv}")
#     print(f"[OK] plot (full)               → {plot_full_path}")
#     if score is not None:
#         print(f"[OK] score                    → {score_path} (score={score:.4f})")
#     else:
#         print(f"[WARN] score 無法計算，請確認 silhouette/corr/d_final 是否可用；已寫入 {score_path}")


# if __name__ == "__main__":
#     main()

# """
# python feature_selection/statistics/t_sne/t_sne.py \
#   --cluster_dir feature_selection/results/ts_kmeans_msm/msm_kmeans_btcusdt_15m_k15 \
#   --ts_csv feature_selection/results/hierarchical_corr/hcorr_pearson_avg_k60/hcorr_pearson_avg_k60_selected_feat.csv \
#   --out_dir feature_selection/results/t_sne \
#   --prefix msm_kmeans_btcusdt_15m_k15 \
#   --msm_cost 0.1 \
#   --use_numba \
#   --perplexity 30 --learning_rate 200 --n_iter 1000 --random_state 0 \
#   --use_tqdm

# """
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

# 載入 MSM 距離函數
# 注意：_msm_distance_nb 必須是已經被 @jit(nopython=True) 裝飾過的函式
from feature_selection.statistics.ts_kmeans_msm.ts_kmeans_msm import _msm_distance_nb, msm_distance

# 嘗試載入 Numba 進行加速
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # 若無 Numba，定義 dummy decorator 以防報錯 (雖然程式邏輯會擋掉)
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# =============================================================================
#  Numba 加速核心區域
# =============================================================================
@jit(nopython=True, parallel=True)
def _compute_msm_matrix_numba(matrix: np.ndarray, c: float) -> np.ndarray:
    """
    使用 Numba parallel (prange) 平行計算矩陣。
    matrix shape: (n_samples, seq_len)
    """
    n = matrix.shape[0]
    # 初始化距離矩陣
    dist = np.zeros((n, n), dtype=np.float32)
    
    # prange 會自動分配迴圈給多個 CPU 核心進行平行運算
    for i in prange(n):
        for j in range(i + 1, n):
            # 呼叫純 Numba 版的 MSM 計算函式
            d = _msm_distance_nb(matrix[i], matrix[j], c)
            dist[i, j] = d
            dist[j, i] = d
            
    return dist
# =============================================================================


def _load_labels(cluster_dir: Path) -> pd.DataFrame:
    label_files = sorted(cluster_dir.glob("*_labels.csv"))
    if not label_files:
        raise FileNotFoundError(f"[t-sne] 找不到 *_labels.csv 於 {cluster_dir}")
    # 如果有多個，優先選擇含資料夾名稱的
    target_name = cluster_dir.name
    chosen = None
    for f in label_files:
        if target_name in f.stem:
            chosen = f
            break
    if chosen is None:
        chosen = label_files[0]
    labels_df = pd.read_csv(chosen)
    if {"id", "cluster"} - set(labels_df.columns):
        raise ValueError("[t-sne] labels 檔案需要包含 id, cluster 欄位")
    return labels_df[["id", "cluster"]]


def _progress(it, enable: bool, desc: str):
    if enable and tqdm is not None:
        return tqdm(it, desc=desc)
    return it


def _load_feature_matrix(ts_csv: Path) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(ts_csv)
    drop_cols = [c for c in ["datetime", "timestamp"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="any")
    if df.empty or df.shape[1] == 0:
        raise ValueError("[t-sne] 時間序列矩陣為空，請檢查輸入 CSV")
    features = list(df.columns)
    matrix = df.to_numpy(dtype=np.float32).T  # shape = (n_features, n_time_steps)
    return matrix, features


def _align_with_clusters(
    matrix: np.ndarray,
    features: List[str],
    labels_df: pd.DataFrame,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    label_map = labels_df.set_index("id")["cluster"]
    available = [f for f in features if f in label_map]
    if not available:
        raise ValueError("[t-sne] 交集為空：feature CSV 與 labels 不對齊")
    missing = [f for f in label_map.index if f not in features]
    if missing:
        print(f"[WARN] {len(missing)} 個在 labels 內但 CSV 中缺少的指標已略過")
    idx_map = {f: i for i, f in enumerate(features)}
    sel_idx = [idx_map[f] for f in available]
    aligned_matrix = matrix[sel_idx]
    clusters = label_map.loc[available].to_numpy(dtype=int)
    return aligned_matrix, available, clusters


def _assign_clusters_all(features: List[str], labels_df: pd.DataFrame) -> np.ndarray:
    label_map = labels_df.set_index("id")["cluster"]
    clusters = np.full(len(features), -1, dtype=int)
    for i, f in enumerate(features):
        if f in label_map:
            clusters[i] = int(label_map[f])
    return clusters


def _infer_corr_path(hcorr_clusters_csv: Path) -> Path:
    stem = hcorr_clusters_csv.stem
    if stem.endswith("_clusters"):
        stem = stem[:-9]
    return hcorr_clusters_csv.with_name(f"{stem}_corr.csv")


def _mean_corr_between_reps(reps: List[str], corr_csv: Path, ts_csv: Path | None = None) -> float | None:
    if len(reps) < 2:
        return 0.0
    df_corr = None
    if corr_csv and corr_csv.exists():
        try:
            df_corr = pd.read_csv(corr_csv, index_col=0)
        except Exception:
            df_corr = None
    if df_corr is None and ts_csv is not None:
        raw = pd.read_csv(ts_csv)
        cols = [c for c in reps if c in raw.columns]
        if not cols:
            return None
        df_corr = raw[cols].corr()
    if df_corr is None:
        return None
    present = [c for c in reps if c in df_corr.index and c in df_corr.columns]
    if len(present) < 2:
        return None
    sub = df_corr.loc[present, present].to_numpy()
    tri = sub[np.triu_indices_from(sub, k=1)]
    if tri.size == 0:
        return 0.0
    return float(np.nanmean(tri))


def _load_silhouette(cluster_dir: Path) -> float | None:
    for f in sorted(cluster_dir.glob("*metrics.json")):
        try:
            data = json.loads(f.read_text())
            if "silhouette_msm" in data:
                return float(data["silhouette_msm"])
        except Exception:
            continue
    return None


def _map_clusters_via_hcorr(
    features: List[str],
    labels_df: pd.DataFrame,
    hcorr_clusters_csv: Path,
    hcorr_reps_csv: Path,
) -> np.ndarray:
    """
    依據 hierarchical corr 60 群 → 代表指標 → MSM 15 群的映射，為所有 feature 指派 MSM 群。
    """
    hc = pd.read_csv(hcorr_clusters_csv)
    reps = pd.read_csv(hcorr_reps_csv)
    if {"feature", "cluster"} - set(hc.columns):
        raise ValueError("[t-sne] hcorr_clusters_csv 需要欄位 feature, cluster")
    if {"cluster_id", "representative"} - set(reps.columns):
        raise ValueError("[t-sne] hcorr_reps_csv 需要欄位 cluster_id, representative")

    feat_to_hcluster = hc.set_index("feature")["cluster"].to_dict()
    hcluster_to_rep = reps.set_index("cluster_id")["representative"].to_dict()
    label_map = labels_df.set_index("id")["cluster"].to_dict()

    clusters = np.full(len(features), -1, dtype=int)
    missing_feat = []
    missing_rep = []
    missing_label = []
    for idx, f in enumerate(features):
        hcid = feat_to_hcluster.get(f)
        if hcid is None:
            missing_feat.append(f)
            continue
        rep = hcluster_to_rep.get(hcid)
        if rep is None:
            missing_rep.append(f)
            continue
        msm_cluster = label_map.get(rep)
        if msm_cluster is None:
            missing_label.append(rep)
            continue
        clusters[idx] = int(msm_cluster)

    if missing_feat:
        print(f"[WARN] {len(missing_feat)} features 在 hcorr_clusters_csv 中找不到，標記為 -1")
    if missing_rep:
        print(f"[WARN] {len(missing_rep)} features 的 hcorr cluster 沒有代表指標，標記為 -1")
    if missing_label:
        uniq_missing = sorted(set(missing_label))
        print(f"[WARN] {len(uniq_missing)} 代表指標沒有對應 MSM 群，標記為 -1: {uniq_missing[:5]}{'...' if len(uniq_missing)>5 else ''}")
    return clusters


def _standardize_rows(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True) + 1e-8
    return (matrix - mean) / std


def _pairwise_msm_distance(matrix: np.ndarray, msm_cost: float, use_numba: bool, use_tqdm: bool) -> np.ndarray:
    """
    計算 pairwise MSM 距離矩陣，matrix shape = (n_features, seq_len)
    """
    # -------------------------------------------------------------
    # 啟用 Numba 平行加速路徑
    # -------------------------------------------------------------
    if use_numba and HAS_NUMBA and _msm_distance_nb is not None:
        if use_tqdm:
            print(f"[INFO] 使用 Numba Parallel 加速計算 MSM 矩陣 (n={matrix.shape[0]}, cost={msm_cost})...")
        # 轉為 float64 確保精度，傳入 numba 編譯過的函式
        return _compute_msm_matrix_numba(matrix.astype(np.float64), msm_cost)
    
    # -------------------------------------------------------------
    # Fallback: 純 Python 迴圈 (如果沒裝 Numba 或 _msm_distance_nb 未定義)
    # -------------------------------------------------------------
    n = matrix.shape[0]
    dist = np.zeros((n, n), dtype=np.float32)
    
    iterator = range(n)
    if use_tqdm and tqdm is not None:
        iterator = tqdm(iterator, desc="MSM pairwise (Python loop)")
        
    for i in iterator:
        dist[i, i] = 0.0
        for j in range(i + 1, n):
            d = msm_distance(matrix[i], matrix[j], c=msm_cost, use_numba=False)
            dist[i, j] = dist[j, i] = d
    return dist


def _run_tsne(
    data: np.ndarray,
    metric: str,
    perplexity: float,
    random_state: int,
    learning_rate: float,
    n_iter: int,
) -> np.ndarray:
    n_samples = data.shape[0]
    max_perp = max(1.0, n_samples - 1.0)
    perp = min(perplexity, max_perp)
    if perp < 2.0 and n_samples > 2:
        perp = min(5.0, max_perp)
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=random_state,
        learning_rate=learning_rate,
        n_iter=n_iter,
        metric=metric,
        init="pca",
    )
    return tsne.fit_transform(data)


def _plot_embedding(
    embedding: np.ndarray, clusters: np.ndarray, out_png: Path, title: str, label_unlabeled: bool = True
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    unique_clusters = sorted(np.unique(clusters).tolist())
    cmap = plt.get_cmap("tab20")
    color_idx = 0
    legend_handles = []
    for cid in unique_clusters:
        mask = clusters == cid
        if cid == -1 and label_unlabeled:
            color = "#888888"
            lbl = f"unlabeled (n={mask.sum()})"
        else:
            color = cmap(color_idx % cmap.N)
            lbl = f"cluster {cid} (n={mask.sum()})"
            color_idx += 1
        pts = ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=28,
            color=color,
            label=lbl,
            alpha=0.85,
            edgecolors="k",
            linewidths=0.2,
        )
        legend_handles.append((pts, lbl))
    ax.set_xlabel("t-SNE X")
    ax.set_ylabel("t-SNE Y")
    ax.set_title(title)
    handles, labels = zip(*legend_handles) if legend_handles else ([], [])
    ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=8,
        framealpha=0.9,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.8, 1.0))
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cluster_dir",
        type=Path,
        required=True,
        help="MSM k-means 結果目錄 (內含 *_labels.csv)",
    )
    ap.add_argument(
        "--ts_csv",
        type=Path,
        required=True,
        help="時間序列特徵 CSV (列為時間步、欄為指標)",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("feature_selection/results/t_sne"),
        help="輸出目錄，預設 feature_selection/results/t_sne",
    )
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--learning_rate", type=float, default=200.0)
    ap.add_argument("--n_iter", type=int, default=1000)
    ap.add_argument("--random_state", type=int, default=0)
    ap.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "msm"],
        help="t-SNE 使用的距離：'euclidean' (快，原本方式) 或 'msm' (需先算 pairwise MSM)",
    )
    ap.add_argument("--msm_cost", type=float, default=0.1, help="MSM 距離的 c 參數")
    ap.add_argument("--use_numba", action="store_true", help="若已安裝 numba，啟用以加速 MSM 距離")
    ap.add_argument("--use_tqdm", action="store_true", help="顯示 tqdm 進度條")
    ap.add_argument(
        "--hier_clusters_csv",
        type=Path,
        default=None,
        help="(選用) hierarchical corr cluster 對應檔，欄位 feature,cluster；若提供將用它 + representatives + MSM labels 為全部 feature 指派群",
    )
    ap.add_argument(
        "--hier_reps_csv",
        type=Path,
        default=None,
        help="(選用) hierarchical corr 代表指標檔，欄位 cluster_id,representative；需搭配 --hier_clusters_csv",
    )
    ap.add_argument("--alpha", type=float, default=1.0, help="score 中的 α，預設 1.0")
    ap.add_argument("--beta", type=float, default=1.0, help="score 中的 β，預設 1.0")
    ap.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="輸出檔名前綴，預設取 cluster_dir 名稱",
    )
    args = ap.parse_args()

    cluster_dir = args.cluster_dir
    ts_csv = args.ts_csv
    out_dir = args.out_dir
    prefix = args.prefix or cluster_dir.name
    run_dir = out_dir / prefix
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.metric == "msm":
        if args.use_tqdm and tqdm is None:
            print("[WARN] tqdm 未安裝，將不顯示進度條")
        if args.use_numba and not HAS_NUMBA:
            print("[WARN] numba 未安裝，MSM 距離將使用純 Python 版本")

    labels_df = _load_labels(cluster_dir)
    matrix, features = _load_feature_matrix(ts_csv)
    matrix = _standardize_rows(matrix)

    reps_list: List[str] | None = None
    if args.hier_clusters_csv and args.hier_reps_csv:
        reps_df = pd.read_csv(args.hier_reps_csv)
        if {"cluster_id", "representative"} - set(reps_df.columns):
            raise ValueError("[t-sne] hier_reps_csv 需包含 cluster_id, representative 欄位")
        reps_list = reps_df["representative"].tolist()
        clusters_full = _map_clusters_via_hcorr(features, labels_df, args.hier_clusters_csv, args.hier_reps_csv)
    elif args.hier_clusters_csv or args.hier_reps_csv:
        raise ValueError("[t-sne] 需同時提供 --hier_clusters_csv 與 --hier_reps_csv，或都不提供")
    else:
        clusters_full = _assign_clusters_all(features, labels_df)
    feature_to_idx = {f: i for i, f in enumerate(features)}
    reps_present: List[str] = []
    if reps_list:
        reps_present = [r for r in reps_list if r in feature_to_idx]
    use_reps = len(reps_present) >= 2

    if use_reps:
        subset_idx = [feature_to_idx[r] for r in reps_present]
        matrix_subset = matrix[subset_idx]
        features_subset = reps_present
        clusters_subset = clusters_full[subset_idx]
        subset_title = "t-SNE of representatives (colored by MSM cluster)"
    else:
        mask_labeled = clusters_full != -1
        if not mask_labeled.any():
            raise ValueError("[t-sne] feature CSV 與 labels 完全無交集，無法上色")
        matrix_subset = matrix[mask_labeled]
        features_subset = [f for f, keep in zip(features, mask_labeled) if keep]
        clusters_subset = clusters_full[mask_labeled]
        subset_title = "t-SNE of labeled features (colored by MSM cluster)"
        subset_idx = np.where(mask_labeled)[0]

    # ================= 核心計算區域 =================
    if args.metric == "msm":
        # 1. 計算所有點的 Pairwise MSM 距離 (可能很慢，所以要加速)
        dist_full = _pairwise_msm_distance(
            matrix,
            msm_cost=args.msm_cost,
            use_numba=args.use_numba,
            use_tqdm=args.use_tqdm and tqdm is not None,
        )
        
        # 2. 準備 t-SNE 的距離矩陣輸入
        if use_reps:
            subset_idx = [feature_to_idx[r] for r in reps_present]
        else:
            subset_idx = np.where(clusters_subset != -1)[0]
        
        # 提取子集合的距離矩陣
        dist_subset = dist_full[np.ix_(subset_idx, subset_idx)]

        # 3. 執行 t-SNE (使用 precomputed)
        embedding = _run_tsne(
            dist_subset,
            metric="precomputed",
            perplexity=args.perplexity,
            random_state=args.random_state,
            learning_rate=args.learning_rate,
            n_iter=args.n_iter,
        )
        embedding_full = _run_tsne(
            dist_full,
            metric="precomputed",
            perplexity=args.perplexity,
            random_state=args.random_state,
            learning_rate=args.learning_rate,
            n_iter=args.n_iter,
        )
    else:
        # 使用歐氏距離 (原本的快速路徑)
        embedding = _run_tsne(
            matrix_subset,
            metric="euclidean",
            perplexity=args.perplexity,
            random_state=args.random_state,
            learning_rate=args.learning_rate,
            n_iter=args.n_iter,
        )
        embedding_full = _run_tsne(
            matrix,
            metric="euclidean",
            perplexity=args.perplexity,
            random_state=args.random_state,
            learning_rate=args.learning_rate,
            n_iter=args.n_iter,
        )

    # ================= 輸出結果區域 =================
    embed_df = pd.DataFrame(
        {
            "feature": features_subset,
            "cluster": clusters_subset.astype(int),
            "tsne_x": embedding[:, 0],
            "tsne_y": embedding[:, 1],
        }
    )
    embed_csv = run_dir / f"{prefix}_tsne_embedding.csv"
    embed_df.to_csv(embed_csv, index=False)

    stats_df = (
        embed_df.groupby("cluster")
        .agg(size=("feature", "count"), tsne_x_mean=("tsne_x", "mean"), tsne_y_mean=("tsne_y", "mean"))
        .reset_index()
        .rename(columns={"cluster": "cluster_id"})
    )
    stats_csv = run_dir / f"{prefix}_tsne_cluster_stats.csv"
    stats_df.to_csv(stats_csv, index=False)

    plot_path = run_dir / f"{prefix}_tsne.png"
    _plot_embedding(
        embedding,
        clusters_subset,
        plot_path,
        title=subset_title,
        label_unlabeled=not use_reps,
    )

    embed_full_df = pd.DataFrame(
        {
            "feature": features,
            "cluster": clusters_full.astype(int),
            "tsne_x": embedding_full[:, 0],
            "tsne_y": embedding_full[:, 1],
        }
    )
    embed_full_csv = run_dir / f"{prefix}_tsne_full_embedding.csv"
    embed_full_df.to_csv(embed_full_csv, index=False)

    stats_full_df = (
        embed_full_df.groupby("cluster")
        .agg(size=("feature", "count"), tsne_x_mean=("tsne_x", "mean"), tsne_y_mean=("tsne_y", "mean"))
        .reset_index()
        .rename(columns={"cluster": "cluster_id"})
    )
    stats_full_csv = run_dir / f"{prefix}_tsne_full_cluster_stats.csv"
    stats_full_df.to_csv(stats_full_csv, index=False)

    plot_full_path = run_dir / f"{prefix}_tsne_full.png"
    _plot_embedding(
        embedding_full,
        clusters_full,
        plot_full_path,
        title="t-SNE of all features (labeled/unlabeled)",
        label_unlabeled=True,
    )

    # 計算分數
    corr_path = _infer_corr_path(args.hier_clusters_csv) if args.hier_clusters_csv else None
    mean_corr = _mean_corr_between_reps(reps_list or [], corr_path, ts_csv if ts_csv.exists() else None)
    silhouette = _load_silhouette(cluster_dir)
    d_final = len(reps_list) if reps_list is not None else int((clusters_full != -1).sum())
    score = None
    if silhouette is not None and mean_corr is not None and d_final > 0:
        score = float(silhouette - args.alpha * mean_corr - args.beta * (d_final / 276.0))
    
    score_path = run_dir / f"{prefix}_tsne_score.json"
    score_payload = {
        "silhouette_msm": silhouette,
        "mean_corr_representatives": mean_corr,
        "d_final": d_final,
        "alpha": args.alpha,
        "beta": args.beta,
        "score": score,
    }
    score_path.write_text(json.dumps(score_payload, indent=2), encoding="utf-8")

    print(f"[OK] t-SNE embedding (labeled) → {embed_csv}")
    print(f"[OK] cluster stats (labeled)   → {stats_csv}")
    print(f"[OK] plot (labeled)            → {plot_path}")
    print(f"[OK] t-SNE embedding (full)    → {embed_full_csv}")
    print(f"[OK] cluster stats (full)      → {stats_full_csv}")
    print(f"[OK] plot (full)               → {plot_full_path}")
    if score is not None:
        print(f"[OK] score                     → {score_path} (score={score:.4f})")
    else:
        print(f"[WARN] score 無法計算，請確認 silhouette/corr/d_final 是否可用；已寫入 {score_path}")


if __name__ == "__main__":
    main()
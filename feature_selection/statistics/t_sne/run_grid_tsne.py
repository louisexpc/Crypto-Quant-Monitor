# from __future__ import annotations

# import argparse
# import subprocess
# import copy
# import tempfile
# from pathlib import Path
# from typing import Iterable, Tuple

# import yaml


# HCORR_KS = [100, 80, 60, 40, 20]
# KMEANS_KS = [40, 30, 20, 15, 10]


# def _run(cmd):
#     return subprocess.run(cmd, capture_output=True, text=True)


# def _load_yaml(path: Path):
#     with open(path, "r", encoding="utf-8") as f:
#         return yaml.safe_load(f)


# def _write_yaml(cfg) -> Path:
#     tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
#     yaml.safe_dump(cfg, tmp)
#     tmp_path = Path(tmp.name)
#     tmp.close()
#     return tmp_path


# def _hcorr_files(base_dir: Path, k: int) -> Tuple[Path, Path, Path] | None:
#     prefix = f"hcorr_pearson_avg_k{k}"
#     folder = base_dir / prefix
#     clusters = folder / f"{prefix}_clusters.csv"
#     reps = folder / f"{prefix}_representatives.csv"
#     selected = folder / f"{prefix}_selected_feat.csv"
#     if clusters.exists() and reps.exists() and selected.exists():
#         return clusters, reps, selected
#     return None


# def _kmeans_prefix(hk: int, kk: int) -> str:
#     return f"msm_kmeans_h{hk}_k{kk}"


# def run_grid(
#     tsne_script: Path,
#     ts_csv: Path,
#     out_dir: Path,
#     hcorr_base: Path,
#     kmeans_base: Path,
#     kmeans_cfg: Path,
#     metric: str = "euclidean",
#     perplexity: float = 30.0,
#     learning_rate: float = 200.0,
#     n_iter: int = 1000,
#     random_state: int = 0,
# ) -> None:
#     out_dir.mkdir(parents=True, exist_ok=True)
#     base_kmeans_cfg = _load_yaml(kmeans_cfg)
#     kmeans_out_dir = Path(base_kmeans_cfg["output"]["dir"]).resolve()

#     for hk in HCORR_KS:
#         hc_files = _hcorr_files(hcorr_base, hk)
#         if hc_files is None:
#             print(f"[SKIP] hcorr k={hk} 檔案不存在")
#             continue
#         hc_clusters, hc_reps, hc_selected = hc_files
#         for kk in KMEANS_KS:
#             k_prefix = _kmeans_prefix(hk, kk)
#             cluster_dir = kmeans_out_dir / k_prefix

#             have_kmeans = cluster_dir.exists()
#             if have_kmeans:
#                 print(f"[SKIP] k-means k={kk} (h={hk}) 已存在，沿用結果")
#             else:
#                 kcfg = copy.deepcopy(base_kmeans_cfg)
#                 kcfg["cluster"]["n_clusters"] = kk
#                 kcfg["cluster"]["use_numba"] = True
#                 kcfg["cluster"]["use_tqdm"] = False
#                 kcfg["input"]["csv_path"] = str(hc_selected)
#                 kcfg["output"]["prefix"] = k_prefix
#                 kcfg_path = _write_yaml(kcfg)
#                 print(f"[RUN] k-means k={kk} (h={hk})")
#                 res = _run(
#                     ["python", "feature_selection/statistics/ts_kmeans_msm/run_ts_kmeans_msm.py", "-c", str(kcfg_path)]
#                 )
#                 if res.returncode != 0:
#                     print(f"[FAIL] k-means k={kk} (h={hk}) returncode={res.returncode}")
#                     print(res.stderr)
#                     continue
#                 have_kmeans = cluster_dir.exists()
#                 if not have_kmeans:
#                     print(f"[SKIP] k-means outputs missing: {cluster_dir}")
#                     continue

#             prefix = f"hcorr_k{hk}_kmeans_k{kk}"
#             cmd = [
#                 "python",
#                 str(tsne_script),
#                 "--cluster_dir",
#                 str(cluster_dir),
#                 "--ts_csv",
#                 str(ts_csv),
#                 "--out_dir",
#                 str(out_dir),
#                 "--prefix",
#                 prefix,
#                 "--metric",
#                 metric,
#                 "--hier_clusters_csv",
#                 str(hc_clusters),
#                 "--hier_reps_csv",
#                 str(hc_reps),
#                 "--perplexity",
#                 str(perplexity),
#                 "--learning_rate",
#                 str(learning_rate),
#                 "--n_iter",
#                 str(n_iter),
#                 "--random_state",
#                 str(random_state),
#             ]
#             print(f"[RUN] {prefix}")
#             res = subprocess.run(cmd, capture_output=True, text=True)
#             if res.returncode != 0:
#                 print(f"[FAIL] {prefix} returncode={res.returncode}")
#                 print(res.stderr)
#             else:
#                 print(f"[OK] {prefix}")


# def parse_args():
#     ap = argparse.ArgumentParser()
#     ap.add_argument(
#         "--ts_csv",
#         type=Path,
#         default=Path("data/precomputed/btcusdt_15m_features_VBT_z_norm.csv"),
#         help="原始 276 特徵矩陣 CSV",
#     )
#     ap.add_argument(
#         "--out_dir",
#         type=Path,
#         default=Path("feature_selection/results/t_sne_grid"),
#         help="t-SNE 輸出根目錄",
#     )
#     ap.add_argument(
#         "--hcorr_base",
#         type=Path,
#         default=Path("feature_selection/results/hierarchical_corr"),
#         help="h-corr 目錄（內含 hcorr_pearson_avg_kXX 子資料夾）",
#     )
#     ap.add_argument(
#         "--kmeans_base",
#         type=Path,
#         default=Path("feature_selection/results/ts_kmeans_msm"),
#         help="MSM k-means 目錄（會以 base config 的 output.dir 為準）",
#     )
#     ap.add_argument(
#         "--kmeans_cfg",
#         type=Path,
#         default=Path("feature_selection/statistics/ts_kmeans_msm/config.yaml"),
#         help="MSM k-means base config（會覆寫 n_clusters/input/output prefix）",
#     )
#     ap.add_argument(
#         "--metric",
#         type=str,
#         default="euclidean",
#         choices=["euclidean", "msm"],
#         help="t-SNE 距離，預設 euclidean；msm 會較慢",
#     )
#     ap.add_argument("--perplexity", type=float, default=30.0)
#     ap.add_argument("--learning_rate", type=float, default=200.0)
#     ap.add_argument("--n_iter", type=int, default=1000)
#     ap.add_argument("--random_state", type=int, default=0)
#     return ap.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     tsne_script = Path(__file__).resolve().parent / "t_sne.py"
#     run_grid(
#         tsne_script=tsne_script,
#         ts_csv=args.ts_csv,
#         out_dir=args.out_dir,
#         hcorr_base=args.hcorr_base,
#         kmeans_base=args.kmeans_base,
#         kmeans_cfg=args.kmeans_cfg,
#         metric=args.metric,
#         perplexity=args.perplexity,
#         learning_rate=args.learning_rate,
#         n_iter=args.n_iter,
#         random_state=args.random_state,
#     )
from __future__ import annotations

import argparse
import subprocess
import copy
import tempfile
import os
from pathlib import Path
from typing import Iterable, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

# ================= 設定區 =================
# 設定同時要跑幾個 Process (建議設為 CPU 核心數 - 2)
MAX_WORKERS = max(1, os.cpu_count() - 2) 

HCORR_KS = [100, 80, 60, 40, 20]
KMEANS_KS = [40, 30, 20, 15, 10]
# =========================================

def _run(cmd):
    # capture_output=True 會把 log 吃掉，若想在並行時除錯，可考慮改為 False 並導向檔案
    return subprocess.run(cmd, capture_output=True, text=True)

def _load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _write_yaml(cfg) -> Path:
    # 為了避免多進程寫入同一個暫存檔，使用 delete=False 並讓每個進程有獨立檔案
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp)
    tmp_path = Path(tmp.name)
    tmp.close()
    return tmp_path

def _hcorr_files(base_dir: Path, k: int) -> Tuple[Path, Path, Path] | None:
    prefix = f"hcorr_pearson_avg_k{k}"
    folder = base_dir / prefix
    clusters = folder / f"{prefix}_clusters.csv"
    reps = folder / f"{prefix}_representatives.csv"
    selected = folder / f"{prefix}_selected_feat.csv"
    if clusters.exists() and reps.exists() and selected.exists():
        return clusters, reps, selected
    return None

def _kmeans_prefix(hk: int, kk: int) -> str:
    return f"msm_kmeans_h{hk}_k{kk}"

# 將原本內層迴圈的邏輯獨立出來，變成一個可以被平行呼叫的任務
def process_single_pair(
    hk: int, 
    kk: int, 
    base_kmeans_cfg: dict, 
    kmeans_out_dir: Path, 
    hc_files: tuple, 
    tsne_script: Path, 
    ts_csv: Path, 
    out_dir: Path, 
    tsne_params: dict
) -> str:
    """
    處理單一 (h, k) 組合：
    1. 檢查/執行 K-Means
    2. 執行 t-SNE
    回傳執行結果字串供主進程列印
    """
    hc_clusters, hc_reps, hc_selected = hc_files
    k_prefix = _kmeans_prefix(hk, kk)
    cluster_dir = kmeans_out_dir / k_prefix

    # --- 1. K-Means 階段 ---
    have_kmeans = cluster_dir.exists()
    if not have_kmeans:
        kcfg = copy.deepcopy(base_kmeans_cfg)
        kcfg["cluster"]["n_clusters"] = kk
        kcfg["cluster"]["use_numba"] = True
        kcfg["cluster"]["use_tqdm"] = False # 平行化時關閉 tqdm 以免 log 混亂
        kcfg["input"]["csv_path"] = str(hc_selected)
        kcfg["output"]["prefix"] = k_prefix
        kcfg_path = _write_yaml(kcfg)
        
        # 呼叫 K-Means script
        res = _run(["python", "feature_selection/statistics/ts_kmeans_msm/run_ts_kmeans_msm.py", "-c", str(kcfg_path)])
        
        # 清理暫存 config
        try:
            os.remove(kcfg_path)
        except OSError:
            pass

        if res.returncode != 0:
            return f"[FAIL][K-Means] h={hk} k={kk}: {res.stderr[:200]}"
            
        if not cluster_dir.exists():
            return f"[FAIL][K-Means] h={hk} k={kk}: Output dir missing"
    
    # --- 2. t-SNE 階段 ---
    prefix = f"hcorr_k{hk}_kmeans_k{kk}"
    cmd = [
        "python", str(tsne_script),
        "--cluster_dir", str(cluster_dir),
        "--ts_csv", str(ts_csv),
        "--out_dir", str(out_dir),
        "--prefix", prefix,
        "--metric", tsne_params["metric"],
        "--hier_clusters_csv", str(hc_clusters),
        "--hier_reps_csv", str(hc_reps),
        "--perplexity", str(tsne_params["perplexity"]),
        "--learning_rate", str(tsne_params["learning_rate"]),
        "--n_iter", str(tsne_params["n_iter"]),
        "--random_state", str(tsne_params["random_state"]),
        "--use_numba" # 強制啟用 numba
    ]
    
    # 若是 msm 且非平行執行，可以考慮開 tqdm，但在 ProcessPool 中建議關閉
    # cmd.append("--use_tqdm") 

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return f"[FAIL][t-SNE] {prefix}: {res.stderr[:200]}"
    
    return f"[OK] {prefix}"

def run_grid(
    tsne_script: Path,
    ts_csv: Path,
    out_dir: Path,
    hcorr_base: Path,
    kmeans_base: Path,
    kmeans_cfg: Path,
    metric: str = "euclidean",
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    n_iter: int = 1000,
    random_state: int = 0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_kmeans_cfg = _load_yaml(kmeans_cfg)
    kmeans_out_dir = Path(base_kmeans_cfg["output"]["dir"]).resolve()

    # 收集參數打包
    tsne_params = {
        "metric": metric,
        "perplexity": perplexity,
        "learning_rate": learning_rate,
        "n_iter": n_iter,
        "random_state": random_state
    }

    tasks = []
    
    # 準備任務列表
    for hk in HCORR_KS:
        hc_files = _hcorr_files(hcorr_base, hk)
        if hc_files is None:
            print(f"[SKIP] hcorr k={hk} 檔案不存在")
            continue
            
        for kk in KMEANS_KS:
            # 將任務加入列表，而不是直接執行
            tasks.append((
                hk, kk, base_kmeans_cfg, kmeans_out_dir, hc_files, 
                tsne_script, ts_csv, out_dir, tsne_params
            ))

    print(f"準備執行 {len(tasks)} 個任務，並行數: {MAX_WORKERS}...")

    # 開始並行執行
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # submit tasks
        futures = [executor.submit(process_single_pair, *task) for task in tasks]
        
        # 獲取結果
        for future in as_completed(futures):
            print(future.result())

def parse_args():
    ap = argparse.ArgumentParser()
    # ... (參數設定與原檔相同，略過不變) ...
    ap.add_argument("--ts_csv", type=Path, default=Path("data/precomputed/btcusdt_15m_features_VBT_z_norm.csv"))
    ap.add_argument("--out_dir", type=Path, default=Path("feature_selection/results/t_sne_grid"))
    ap.add_argument("--hcorr_base", type=Path, default=Path("feature_selection/results/hierarchical_corr"))
    ap.add_argument("--kmeans_base", type=Path, default=Path("feature_selection/results/ts_kmeans_msm"))
    ap.add_argument("--kmeans_cfg", type=Path, default=Path("feature_selection/statistics/ts_kmeans_msm/config.yaml"))
    ap.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "msm"])
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--learning_rate", type=float, default=200.0)
    ap.add_argument("--n_iter", type=int, default=1000)
    ap.add_argument("--random_state", type=int, default=0)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    tsne_script = Path(__file__).resolve().parent / "t_sne.py"
    run_grid(
        tsne_script=tsne_script,
        ts_csv=args.ts_csv,
        out_dir=args.out_dir,
        hcorr_base=args.hcorr_base,
        kmeans_base=args.kmeans_base,
        kmeans_cfg=args.kmeans_cfg,
        metric=args.metric,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        random_state=args.random_state,
    )
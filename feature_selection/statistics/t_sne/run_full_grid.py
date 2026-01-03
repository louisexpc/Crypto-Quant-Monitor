# from __future__ import annotations

# import argparse
# import copy
# import os
# import subprocess
# import tempfile
# from pathlib import Path
# from typing import Dict, List, Tuple

# import yaml

# try:
#     from tqdm.auto import tqdm as _tqdm
# except Exception:
#     def _tqdm(x, **kwargs):
#         return x

# HCORR_KS = [100, 80, 60, 40, 20]
# KMEANS_KS = [40, 30, 20, 15, 10]


# def _run(cmd: List[str], env: Dict[str, str] | None = None, capture: bool = True) -> Tuple[int, str, str]:
#     run_env = os.environ.copy()
#     if env:
#         run_env.update(env)
#     if capture:
#         res = subprocess.run(cmd, capture_output=True, text=True, env=run_env)
#         return res.returncode, res.stdout, res.stderr
#     res = subprocess.run(cmd, env=run_env)
#     return res.returncode, "", ""


# def _load_yaml(path: Path) -> Dict:
#     with open(path, "r", encoding="utf-8") as f:
#         return yaml.safe_load(f)


# def _write_yaml(cfg: Dict) -> Path:
#     tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
#     yaml.safe_dump(cfg, tmp)
#     tmp_path = Path(tmp.name)
#     tmp.close()
#     return tmp_path


# def _ensure_paths(hcorr_cfg: Dict, kmeans_cfg: Dict) -> Tuple[Path, Path, Path, Path]:
#     ts_csv = Path(hcorr_cfg["input"]["csv_path"]).resolve()
#     hcorr_out_dir = Path(hcorr_cfg["output"]["dir"]).resolve()
#     kmeans_out_dir = Path(kmeans_cfg["output"]["dir"]).resolve()
#     tsne_out_dir = Path("feature_selection/results/t_sne_grid").resolve()
#     return ts_csv, hcorr_out_dir, kmeans_out_dir, tsne_out_dir


# def _hcorr_prefix(k: int) -> str:
#     return f"hcorr_pearson_avg_k{k}"


# def _kmeans_prefix(hk: int, kk: int) -> str:
#     return f"msm_kmeans_h{hk}_k{kk}"


# def run_grid(
#     hcorr_cfg_path: Path,
#     kmeans_cfg_path: Path,
#     tsne_script: Path,
#     metric: str,
#     perplexity: float,
#     learning_rate: float,
#     n_iter: int,
#     random_state: int,
#     alpha: float,
#     beta: float,
# ) -> None:
#     base_hcorr_cfg = _load_yaml(hcorr_cfg_path)
#     base_kmeans_cfg = _load_yaml(kmeans_cfg_path)
#     ts_csv, hcorr_out_dir, kmeans_out_dir, tsne_out_dir = _ensure_paths(base_hcorr_cfg, base_kmeans_cfg)

#     tsne_script = tsne_script.resolve()
#     for hk in HCORR_KS:
#         h_prefix = _hcorr_prefix(hk)
#         h_run_dir = hcorr_out_dir / h_prefix
#         clusters_csv = h_run_dir / f"{h_prefix}_clusters.csv"
#         reps_csv = h_run_dir / f"{h_prefix}_representatives.csv"
#         selected_feat_csv = h_run_dir / f"{h_prefix}_selected_feat.csv"
#         have_hcorr = clusters_csv.exists() and reps_csv.exists() and selected_feat_csv.exists()
#         hcfg = copy.deepcopy(base_hcorr_cfg)
#         hcfg["cluster"]["n_clusters"] = hk
#         hcfg["output"]["prefix"] = h_prefix
#         hcfg_path = _write_yaml(hcfg)

#         if have_hcorr:
#             print(f"[SKIP] h-corr k={hk} 已存在，沿用結果")
#         else:
#             print(f"[RUN] h-corr k={hk}")
#             rc, out, err = _run(
#                 ["python", "feature_selection/statistics/hierarchical_corr/run_hcorr.py", "-c", str(hcfg_path)]
#             )
#             if rc != 0:
#                 print(f"[FAIL] h-corr k={hk}: {err}")
#                 continue
#             have_hcorr = clusters_csv.exists() and reps_csv.exists() and selected_feat_csv.exists()
#             if not have_hcorr:
#                 print(f"[SKIP] h-corr outputs missing for k={hk}")
#                 continue

#         for kk in _tqdm(KMEANS_KS, desc=f"k-means (h={hk})", leave=False):
#             k_prefix = _kmeans_prefix(hk, kk)
#             cluster_dir = kmeans_out_dir / k_prefix
#             labels_csv = cluster_dir / f"{k_prefix}_labels.csv"
#             have_kmeans = labels_csv.exists()
#             kcfg = copy.deepcopy(base_kmeans_cfg)
#             kcfg["cluster"]["n_clusters"] = kk
#             kcfg["input"]["csv_path"] = str(selected_feat_csv)
#             kcfg["output"]["prefix"] = k_prefix
#             kcfg_path = _write_yaml(kcfg)

#             kmeans_ran = False
#             if have_kmeans:
#                 print(f"[SKIP] k-means k={kk} (h={hk}) 已存在，沿用結果")
#             else:
#                 print(f"[RUN] k-means k={kk} (h={hk})")
#                 rc, out, err = _run(
#                     ["bash", "feature_selection/statistics/ts_kmeans_msm/run.sh"],
#                     env={"CFG": str(kcfg_path)},
#                     capture=False,  # passthrough to show inner tqdm
#                 )
#                 if rc != 0:
#                     print(f"[FAIL] k-means k={kk} (h={hk}): {err}")
#                     continue
#                 kmeans_ran = True
#                 have_kmeans = cluster_dir.exists()
#                 if not have_kmeans:
#                     print(f"[SKIP] cluster_dir missing: {cluster_dir}")
#                     continue

#             tsne_prefix = f"{h_prefix}__{k_prefix}"
#             metrics_path = cluster_dir / f"{k_prefix}_metrics.json"
#             if not metrics_path.exists() and not kmeans_ran:
#                 print(f"[RUN] reporter for k-means k={kk} (h={hk})")
#                 _run(
#                     [
#                         "python",
#                         "feature_selection/statistics/ts_kmeans_msm/reporter.py",
#                         "-c",
#                         str(kcfg_path),
#                         "--dir",
#                         str(kmeans_out_dir),
#                         "--prefix",
#                         k_prefix,
#                     ]
#                 )
#             cmd = [
#                 "python",
#                 str(tsne_script),
#                 "--cluster_dir",
#                 str(cluster_dir),
#                 "--ts_csv",
#                 str(ts_csv),
#                 "--out_dir",
#                 str(tsne_out_dir),
#                 "--prefix",
#                 tsne_prefix,
#                 "--metric",
#                 metric,
#                 "--hier_clusters_csv",
#                 str(clusters_csv),
#                 "--hier_reps_csv",
#                 str(reps_csv),
#                 "--perplexity",
#                 str(perplexity),
#                 "--learning_rate",
#                 str(learning_rate),
#                 "--n_iter",
#                 str(n_iter),
#                 "--random_state",
#                 str(random_state),
#                 "--alpha",
#                 str(alpha),
#                 "--beta",
#                 str(beta),
#             ]
#             print(f"[RUN] t-SNE ({tsne_prefix})")
#             rc, out, err = _run(cmd, capture=False)  # passthrough to show t-SNE tqdm/logs
#             if rc != 0:
#                 print(f"[FAIL] t-SNE ({tsne_prefix}): {err}")
#             else:
#                 print(f"[OK] {tsne_prefix}")


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument(
#         "--hcorr_cfg",
#         type=Path,
#         default=Path("feature_selection/statistics/hierarchical_corr/config.yaml"),
#         help="base h-corr config",
#     )
#     ap.add_argument(
#         "--kmeans_cfg",
#         type=Path,
#         default=Path("feature_selection/statistics/ts_kmeans_msm/config.yaml"),
#         help="base MSM k-means config",
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
#     ap.add_argument("--alpha", type=float, default=1.0, help="score 公式中的 α")
#     ap.add_argument("--beta", type=float, default=1.0, help="score 公式中的 β")
#     args = ap.parse_args()

#     tsne_script = Path(__file__).parent / "t_sne.py"
#     run_grid(
#         hcorr_cfg_path=args.hcorr_cfg,
#         kmeans_cfg_path=args.kmeans_cfg,
#         tsne_script=tsne_script,
#         metric=args.metric,
#         perplexity=args.perplexity,
#         learning_rate=args.learning_rate,
#         n_iter=args.n_iter,
#         random_state=args.random_state,
#         alpha=args.alpha,
#         beta=args.beta,
#     )


# if __name__ == "__main__":
#     main()

# """
# python feature_selection/statistics/t_sne/run_full_grid.py \
#   --hcorr_cfg feature_selection/statistics/hierarchical_corr/config.yaml \
#   --kmeans_cfg feature_selection/statistics/ts_kmeans_msm/config.yaml \
#   --metric msm \
#   --alpha 0.5 --beta 0.2 \
#   --perplexity 30 --learning_rate 200 \
#   --n_iter 1000 --random_state 0
# """
from __future__ import annotations

import argparse
import subprocess
import copy
import tempfile
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

# =================【關鍵修改區：防止 WSL 當機】=================

# 1. 強制設定最大同時執行數 (建議設為 2~4)
#    如果您有 32GB RAM，設 4；如果有 16GB RAM，設 2。
MAX_WORKERS = 8  

# 2. 限制每個 Process 內部的 Numba/OpenMP 執行緒數
#    因為我們已經在外層做平行了，內層就不要再搶資源，設為 1 最穩定
ENV_Thread_LIMIT = "1"

# =============================================================

HCORR_KS = [100, 80, 60, 40, 20]
KMEANS_KS = [40, 30, 20, 15, 10]


def _run(cmd: List[str], env_vars: dict = None) -> subprocess.CompletedProcess:
    # 複製當前環境變數
    run_env = os.environ.copy()
    if env_vars:
        run_env.update(env_vars)
        
    # 為了避免並行時 stdout 混亂，預設 capture_output=True
    return subprocess.run(cmd, capture_output=True, text=True, env=run_env)


def _load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(cfg) -> Path:
    # delete=False 是必須的，因為 subprocess 需要讀取這個檔案
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp)
    tmp_path = Path(tmp.name)
    tmp.close()
    return tmp_path


def _hcorr_files(base_dir: Path, k: int) -> Optional[Tuple[Path, Path, Path]]:
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


def process_single_pair(
    hk: int, 
    kk: int, 
    base_kmeans_cfg: dict, 
    kmeans_out_dir: Path, 
    hc_files: tuple, 
    tsne_script: Path, 
    ts_csv: Path, 
    out_dir: Path, 
    tsne_params: dict,
    thread_limit: str
) -> str:
    """
    [Worker Function] 處理單一 (h, k) 組合
    """
    # 設定環境變數以限制內層執行緒 (防止 CPU/RAM 爆炸)
    worker_env = {
        "OMP_NUM_THREADS": thread_limit,
        "OPENBLAS_NUM_THREADS": thread_limit,
        "MKL_NUM_THREADS": thread_limit,
        "VECLIB_MAXIMUM_THREADS": thread_limit,
        "NUMEXPR_NUM_THREADS": thread_limit,
        "NUMBA_NUM_THREADS": thread_limit,
    }

    hc_clusters, hc_reps, hc_selected = hc_files
    k_prefix = _kmeans_prefix(hk, kk)
    cluster_dir = kmeans_out_dir / k_prefix

    # ---------------------------------------------------------
    # 1. K-Means 階段
    # ---------------------------------------------------------
    have_kmeans = cluster_dir.exists()
    
    if not have_kmeans:
        kcfg = copy.deepcopy(base_kmeans_cfg)
        kcfg["cluster"]["n_clusters"] = kk
        kcfg["cluster"]["use_numba"] = True   
        kcfg["cluster"]["use_tqdm"] = False   
        kcfg["input"]["csv_path"] = str(hc_selected)
        kcfg["output"]["prefix"] = k_prefix
        
        kcfg_path = _write_yaml(kcfg)
        
        cmd_kmeans = [
            "python", 
            "feature_selection/statistics/ts_kmeans_msm/run_ts_kmeans_msm.py", 
            "-c", str(kcfg_path)
        ]
        # 傳入限制執行緒的 env
        res = _run(cmd_kmeans, env_vars=worker_env)
        
        try:
            os.remove(kcfg_path)
        except OSError:
            pass

        if res.returncode != 0:
            return f"[FAIL][K-Means] h={hk} k={kk}: {res.stderr[:200].replace(os.linesep, ' ')}"
            
        if not cluster_dir.exists():
            return f"[FAIL][K-Means] h={hk} k={kk}: Output dir missing after run"
    
    # ---------------------------------------------------------
    # 2. t-SNE 階段
    # ---------------------------------------------------------
    prefix = f"hcorr_k{hk}_kmeans_k{kk}"
    
    cmd_tsne = [
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
        "--use_numba"
    ]
    
    if "msm_cost" in tsne_params:
        cmd_tsne.extend(["--msm_cost", str(tsne_params["msm_cost"])])

    # 傳入限制執行緒的 env
    res = _run(cmd_tsne, env_vars=worker_env)
    
    if res.returncode != 0:
        return f"[FAIL][t-SNE] {prefix}: {res.stderr[:200].replace(os.linesep, ' ')}"
    
    status = "Computed" if not have_kmeans else "KMeans-Cached"
    return f"[OK] {prefix} ({status})"


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
    msm_cost: float = 0.1,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_kmeans_cfg = _load_yaml(kmeans_cfg)
    kmeans_out_dir = Path(base_kmeans_cfg["output"]["dir"]).resolve()

    tsne_params = {
        "metric": metric,
        "perplexity": perplexity,
        "learning_rate": learning_rate,
        "n_iter": n_iter,
        "random_state": random_state,
        "msm_cost": msm_cost
    }

    tasks = []
    
    print(f"Checking H-Corr files in: {hcorr_base}")
    
    for hk in HCORR_KS:
        hc_files = _hcorr_files(hcorr_base, hk)
        if hc_files is None:
            print(f"[SKIP] hcorr k={hk} 檔案不存在")
            continue
            
        for kk in KMEANS_KS:
            tasks.append((
                hk, kk, base_kmeans_cfg, kmeans_out_dir, hc_files, 
                tsne_script, ts_csv, out_dir, tsne_params, ENV_Thread_LIMIT
            ))

    total_tasks = len(tasks)
    print(f"準備執行 {total_tasks} 個任務")
    print(f"安全模式: 並行數 MAX_WORKERS={MAX_WORKERS}, 內層線程限制={ENV_Thread_LIMIT}")
    print("-" * 60)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_pair, *task) for task in tasks]
        
        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            result_msg = future.result()
            print(f"[{completed_count}/{total_tasks}] {result_msg}")


def parse_args():
    ap = argparse.ArgumentParser()
    # ... (參數與之前相同，略) ...
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
    ap.add_argument("--msm_cost", type=float, default=0.1)
    
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tsne_script = Path(__file__).resolve().parent / "t_sne.py"
    
    if not tsne_script.exists():
        print(f"[Error] 找不到 t_sne.py: {tsne_script}")
        sys.exit(1)

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
        msm_cost=args.msm_cost,
    )
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _parse_folder_name(name: str) -> Optional[Tuple[int, int]]:
    """
    支援格式: hcorr_pearson_avg_k{m}__msm_kmeans_h{m}_k{n}
    回傳 (m, n)；若不符則回傳 None
    """
    m = re.match(r"hcorr_pearson_avg_k(?P<m>\d+)__msm_kmeans_h\d+_k(?P<n>\d+)", name)
    if not m:
        return None
    try:
        return int(m.group("m")), int(m.group("n"))
    except Exception:
        return None


def _load_score_json(folder: Path) -> Optional[Dict]:
    name = folder.name
    path = folder / f"{name}_tsne_score.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _load_metrics(kmeans_base: Path, m: int, n: int) -> Dict:
    k_prefix = f"msm_kmeans_h{m}_k{n}"
    metrics_path = kmeans_base / k_prefix / f"{k_prefix}_metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text())
    except Exception:
        return {}


def build_summary(root: Path, kmeans_base: Path) -> List[Dict]:
    rows: List[Dict] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        parsed = _parse_folder_name(sub.name)
        if parsed is None:
            continue
        m, n = parsed
        score_data = _load_score_json(sub)
        if not score_data:
            print(f"[WARN] score json missing/unreadable: {sub}")
            continue
        metrics = _load_metrics(kmeans_base, m, n)
        rows.append(
            {
                "m": m,
                "n": n,
                "silhouette_msm": score_data.get("silhouette_msm"),
                "cluster_silhouette_mean": metrics.get("cluster_silhouette_mean"),
                "cluster_intra_mean_avg": metrics.get("cluster_intra_mean_avg"),
                "cluster_inter_mean_avg": metrics.get("cluster_inter_mean_avg"),
                "mean_corr_representatives": score_data.get("mean_corr_representatives"),
                "d_final": score_data.get("d_final"),
                "score": score_data.get("score"),
            }
        )
    rows.sort(key=lambda r: (r["m"], r["n"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("feature_selection/results/t_sne_grid"),
        help="t-sne grid 輸出根目錄",
    )
    ap.add_argument(
        "--out_csv",
        type=Path,
        default=None,
        help="summary 輸出檔；未指定時寫入 root/summary.csv",
    )
    ap.add_argument(
        "--kmeans_base",
        type=Path,
        default=Path("feature_selection/results/ts_kmeans_msm"),
        help="k-means 結果根目錄（讀取 metrics.json 用）",
    )
    args = ap.parse_args()

    root = args.root
    out_csv = args.out_csv or (root / "summary.csv")
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")

    rows = build_summary(root, args.kmeans_base)
    if not rows:
        print("[WARN] no entries found; summary not written")
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "m",
                "n",
                "silhouette_msm",
                "cluster_silhouette_mean",
                "cluster_intra_mean_avg",
                "cluster_inter_mean_avg",
                "mean_corr_representatives",
                "d_final",
                "score",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[OK] summary → {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
"""

python feature_selection/statistics/t_sne/reporter.py \
  --root feature_selection/results/t_sne_grid \
  --kmeans_base feature_selection/results/ts_kmeans_msm \
  --out_csv feature_selection/results/t_sne_grid/summary.csv
"""
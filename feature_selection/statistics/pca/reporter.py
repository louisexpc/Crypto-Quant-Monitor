#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_meta(meta_path: Path) -> Dict:
    """ 1. 說明: 讀取 run_pca 產生的 meta.json
        2. inputs: meta_path: 檔案路徑
        3. return: dict 內容 """
    return json.loads(meta_path.read_text(encoding="utf-8"))


def plot_fold_scree(evr_matrix: List[List[float]], outdir: Path) -> None:
    """ 1. 說明: 每折的 Scree Plot（解釋變異比）與平均曲線
        2. inputs: evr_matrix: 每折的 explained_variance_ratio_ 清單
                   outdir: 圖檔輸出資料夾
        3. return: None（存檔） """
    outdir.mkdir(parents=True, exist_ok=True)
    max_len = max(len(e) for e in evr_matrix)
    plt.figure()
    for i, evr in enumerate(evr_matrix, start=1):
        x = list(range(1, len(evr) + 1))
        plt.plot(x, evr, marker=".", alpha=0.4, label=f"fold{i}")
    # 平均線
    pad = np.full((len(evr_matrix), max_len), np.nan)
    for r, e in enumerate(evr_matrix):
        pad[r, : len(e)] = np.array(e)
    mean_evr = np.nanmean(pad, axis=0)
    x_mean = list(range(1, len(mean_evr) + 1))
    plt.plot(x_mean, mean_evr, marker="o", linewidth=2.0, label="mean")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Scree Plot by Fold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "scree_by_fold.png", dpi=150)
    plt.close()


def summarize(meta: Dict, outdir: Path) -> None:
    """ 1. 說明: 匯總重點—各折成分數、累積解釋率、平均 EVR
        2. inputs: meta: run_pca 的 meta.json 內容
                   outdir: 輸出資料夾
        3. return: None（存檔） """
    outdir.mkdir(parents=True, exist_ok=True)
    df_summary = pd.DataFrame(meta["fold_summaries"])
    df_summary.to_csv(outdir / "pca_summary.csv", index=False)

    # 平均 EVR 與累積
    evr_matrix = meta["evr_matrix"]
    max_len = max(len(e) for e in evr_matrix)
    pad = np.full((len(evr_matrix), max_len), np.nan)
    for r, e in enumerate(evr_matrix):
        pad[r, : len(e)] = np.array(e)
    mean_evr = np.nanmean(pad, axis=0)
    df_mean = pd.DataFrame(
        {"pc": [f"PC{j+1}" for j in range(len(mean_evr))], "mean_explained_variance_ratio": mean_evr}
    )
    df_mean["mean_cumvar"] = df_mean["mean_explained_variance_ratio"].cumsum()
    df_mean.to_csv(outdir / "mean_evr.csv", index=False)

    # 輸出文字摘要
    msg = (
        f"折數: {len(evr_matrix)} | 平均成分數: ~{int(round(df_summary['n_components'].mean()))} | "
        f"平均累積解釋率: {df_summary['cum_explained_variance'].mean():.4f}"
    )
    (outdir / "REPORT.txt").write_text(msg, encoding="utf-8")


def main() -> None:
    """ 1. 說明: 入口點—讀取 meta.json，產生 scree_by_fold.png / pca_summary.csv / mean_evr.csv / REPORT.txt
        2. inputs: 無（命令列參數讀取）
        3. return: None """
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=str, required=True, help="例如 runs/pca/meta.json")
    ap.add_argument("--outdir", type=str, default=None, help="報表輸出資料夾；預設用 meta.json 的資料夾/report")
    args = ap.parse_args()

    meta_path = Path(args.meta)
    meta = load_meta(meta_path)
    outdir = Path(args.outdir) if args.outdir else meta_path.parent / "report"
    outdir.mkdir(parents=True, exist_ok=True)

    summarize(meta, outdir)
    plot_fold_scree(meta["evr_matrix"], outdir)
    print(f"[Reporter] 完成，輸出在：{outdir}")


if __name__ == "__main__":
    main()

"""
python feature_selection/statistics/pca/reporter.py --meta feature_selection/results/pca/meta.json
"""

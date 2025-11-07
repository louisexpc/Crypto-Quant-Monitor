#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_meta(meta_path: Path) -> Dict:
    """ 1. 說明: 讀取 run_umap 產生的 meta.json
        2. inputs: meta_path: 檔案路徑
        3. return: dict 內容 """
    return json.loads(meta_path.read_text(encoding="utf-8"))


def scatter_by_fold(df: pd.DataFrame, datetime_col: str, outdir: Path) -> None:
    """ 1. 說明: 依 fold 上色的 UMAP 2D 散點（用來檢查各折的對齊感）
        2. inputs: df: 含 [datetime, UMAP1, UMAP2, fold], datetime_col: 名稱, outdir: 輸出資料夾
        3. return: None（直接存檔） """
    if "UMAP1" not in df.columns or "UMAP2" not in df.columns:
        return
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(df["UMAP1"], df["UMAP2"], c=df["fold"], s=4, cmap="tab10", alpha=0.85, edgecolors="none")
    plt.colorbar(sc, label="fold")
    plt.xlabel("UMAP1"); plt.ylabel("UMAP2"); plt.title("UMAP (colored by fold)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "umap_scatter_by_fold.png", dpi=150)
    plt.close()


def timeseries(df: pd.DataFrame, datetime_col: str, outdir: Path) -> None:
    """ 1. 說明: UMAP1/UMAP2 的時間序列圖（檢查 regime 轉換）
        2. inputs: df: 含 [datetime, UMAP1, UMAP2], datetime_col: 名稱, outdir: 輸出資料夾
        3. return: None（直接存檔） """
    if "UMAP1" not in df.columns or "UMAP2" not in df.columns:
        return
    df = df.sort_values(datetime_col)
    plt.figure(figsize=(10, 4))
    plt.plot(df[datetime_col], df["UMAP1"], linewidth=0.8, label="UMAP1")
    plt.plot(df[datetime_col], df["UMAP2"], linewidth=0.8, label="UMAP2")
    plt.title("UMAP components over time")
    plt.xlabel("time"); plt.ylabel("value")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "umap_timeseries.png", dpi=150)
    plt.close()


def main() -> None:
    """ 1. 說明: 入口點—讀 meta.json 與 umap_output_oos.csv，輸出可視化
        2. inputs: 命令列參數 --meta, --csv
        3. return: None """
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=str, required=True, help="例如 runs/umap/meta.json")
    ap.add_argument("--csv", type=str, default=None, help="覆蓋用：指定 umap_output_oos.csv 路徑")
    args = ap.parse_args()

    meta = load_meta(Path(args.meta))
    outdir = Path(args.meta).parent / "report"
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv) if args.csv else Path(args.meta).parent / "umap_output_oos.csv"
    df = pd.read_csv(csv_path)
    dtcol = meta["config"]["datetime_col"]

    # 圖 1：依時間漸層的散點（run_umap 已輸出）
    # 圖 2：依 fold 上色
    scatter_by_fold(df, dtcol, outdir)
    # 圖 3：UMAP 分量時間序列
    timeseries(df, dtcol, outdir)

    print(f"[Reporter] 完成，輸出在：{outdir}")


if __name__ == "__main__":
    main()

"""
python feature_selection/statistics/umap/reporter.py --meta feature_selection/results/umap/meta.json
"""

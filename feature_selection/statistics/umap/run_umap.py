#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from feature_selection.statistics.common.utils import (
    load_yaml,
    ensure_dir,
    build_purged_kfold_indices,
    find_pc_columns,
)


try:
    # umap-learn
    from umap import UMAP
except Exception as e:
    raise ImportError("需要安裝 umap-learn：pip install umap-learn") from e


# ----------------------------- Config -----------------------------

@dataclass
class UMAPConfig:
    """ 1. 說明: UMAP 執行所需的 YAML 設定封裝
        2. inputs: 由命令列讀取後的 dict（見 umap_config.yaml）
        3. return: 作為其他函式的設定物件 """
    input_pca_csv: str
    output_dir: str
    datetime_col: str
    pc_prefix: str
    embedding_dim: int
    n_neighbors: int
    min_dist: float
    metric: str
    random_state: int
    n_splits: int
    embargo_minutes: int
    shuffle: bool
    plot_scatter: bool
    max_points: int
    point_size: int


def fit_umap(X_train: np.ndarray, cfg: UMAPConfig) -> UMAP:
    """ 1. 說明: 用訓練折資料 fit 一個 UMAP 模型
        2. inputs: X_train: (n_train, k) 訓練折的 PC 特徵矩陣, cfg: 設定
        3. return: 已訓練好的 UMAP 物件 """
    um = UMAP(
        n_neighbors=cfg.n_neighbors,
        n_components=cfg.embedding_dim,
        min_dist=cfg.min_dist,
        metric=cfg.metric,
        random_state=cfg.random_state,
        verbose=False,
    )
    um.fit(X_train)
    return um


def plot_umap_scatter(df_embed: pd.DataFrame, cfg: UMAPConfig, outdir: Path) -> None:
    """ 1. 說明: 畫 UMAP 2D 散點圖（顏色用時間順序）
        2. inputs: df_embed: 含 [datetime_col, UMAP1..m], cfg: 設定, outdir: 圖檔輸出資料夾
        3. return: None（直接存檔） """
    if cfg.embedding_dim < 2:
        return
    if not cfg.plot_scatter:
        return

    # 時間排序索引轉成顏色
    order = df_embed[cfg.datetime_col].rank(method="first").to_numpy()
    order = (order - order.min()) / (order.max() - order.min() + 1e-9)

    # 下採樣太多點以免圖檔過大（不影響 CSV）
    if len(df_embed) > cfg.max_points:
        idx = np.linspace(0, len(df_embed) - 1, cfg.max_points).astype(int)
        sub = df_embed.iloc[idx].copy()
        order = order[idx]
    else:
        sub = df_embed

    plt.figure(figsize=(6, 5))
    plt.scatter(sub["UMAP1"], sub["UMAP2"], c=order, s=cfg.point_size, cmap="viridis", alpha=0.8, edgecolors="none")
    plt.xlabel("UMAP1"); plt.ylabel("UMAP2"); plt.title("UMAP embedding (color = time order)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "umap_scatter.png", dpi=150)
    plt.close()


# ----------------------------- Main Runner -----------------------------

def run(cfg: UMAPConfig) -> None:
    """ 1. 說明: 主流程—讀取 PCA 分數→建立時序切分→每折 fit UMAP（train）→ transform（test）→ 輸出 OOS UMAP
        2. inputs: cfg: UMAPConfig
        3. return: None（在輸出資料夾寫 CSV/圖與 meta.json） """
    outdir = ensure_dir(cfg.output_dir)

    # 讀 pca_output.csv
    df = pd.read_csv(cfg.input_pca_csv)
    if cfg.datetime_col not in df.columns:
        raise ValueError(f"找不到時間欄位 {cfg.datetime_col}")
    df[cfg.datetime_col] = pd.to_datetime(df[cfg.datetime_col], utc=True)
    df.sort_values(cfg.datetime_col, inplace=True, ignore_index=True)

    pc_cols = find_pc_columns(df, cfg.pc_prefix)
    if not pc_cols:
        raise ValueError("未找到任何 PC 欄位（例如 PC1..PCk）。請確認 pc_prefix 與輸入檔。")

    X = df[pc_cols].to_numpy(dtype=float)
    times = pd.DatetimeIndex(df[cfg.datetime_col])

    # Purged K-Fold 切分
    splits = build_purged_kfold_indices(times, cfg.n_splits, cfg.embargo_minutes, cfg.shuffle)

    # 每筆樣本唯一的 fold id（以其身為 test 的那一折為準）
    fold_id = np.zeros(len(df), dtype=int)
    for i, (_, te_idx) in enumerate(splits, start=1):
        fold_id[te_idx] = i

    # OOS：僅 transform 測試折（每筆只被 transform 一次、使用不包含本身的模型）
    emb_dim = cfg.embedding_dim
    E = np.full((len(df), emb_dim), np.nan, dtype=float)

    fold_summaries = []
    for i, (tr_idx, te_idx) in enumerate(splits, start=1):
        um = fit_umap(X[tr_idx, :], cfg)
        E[te_idx, :] = um.transform(X[te_idx, :])

        fold_summaries.append({
            "fold": i,
            "train_size": int(len(tr_idx)),
            "test_size": int(len(te_idx)),
            "n_neighbors": cfg.n_neighbors,
            "min_dist": cfg.min_dist,
            "metric": cfg.metric
        })

    # 輸出 OOS 嵌入
    embed_cols = [f"UMAP{j+1}" for j in range(emb_dim)]
    out_df = pd.concat(
        [df[[cfg.datetime_col]].reset_index(drop=True),
         pd.DataFrame(E, columns=embed_cols),
         pd.DataFrame({"fold": fold_id})],
        axis=1
    )
    out_df.to_csv(outdir / "umap_output_oos.csv", index=False)

    # Meta 與簡要報告
    meta = {
        "config": cfg.__dict__,
        "pc_cols": pc_cols,
        "embedding_cols": embed_cols,
        "fold_summaries": fold_summaries
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 視覺化（2D 才畫）
    if cfg.embedding_dim >= 2 and cfg.plot_scatter:
        plot_umap_scatter(out_df[[cfg.datetime_col] + embed_cols], cfg, outdir)

    print(f"[UMAP] 完成，OOS 嵌入已輸出：{outdir/'umap_output_oos.csv'}")


def parse_args() -> argparse.Namespace:
    """ 1. 說明: 解析命令列參數
        2. inputs: 無（直接讀 sys.argv）
        3. return: argparse.Namespace """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="umap_config.yaml 路徑")
    return ap.parse_args()


def main() -> None:
    """ 1. 說明: 入口點—讀設定 → 執行 run()
        2. inputs: 無
        3. return: None """
    args = parse_args()
    raw = load_yaml(args.config)
    cfg = UMAPConfig(**raw)
    run(cfg)


if __name__ == "__main__":
    main()

"""
python feature_selection/statistics/umap/run_umap.py --config feature_selection/statistics/umap/umap_config.yaml
"""

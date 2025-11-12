

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 共用工具：從這邊引入
from feature_selection.statistics.common.utils import (
    load_yaml,
    ensure_dir,
    build_purged_kfold_indices,
    select_feature_columns,
)


# ----------------------------- Config -----------------------------

@dataclass
class PCAConfig:
    """ 1. 說明: PCA 執行所需的 YAML 設定封裝（已無 label_cols）
        2. inputs: 由命令列傳入 YAML 解析後的 dict
        3. return: 作為其他函式的設定物件 """
    input_csv: str
    output_dir: str
    datetime_col: str
    exclude_patterns: List[str]
    n_components: float | int
    whiten: bool
    impute: str
    n_splits: int
    embargo_minutes: int
    shuffle: bool
    topn_loadings: int
    random_state: int


# ----------------------------- Helper (PCA 專屬) -----------------------------

def make_pipeline(n_components: float | int, whiten: bool, impute: str) -> Pipeline:
    """ 1. 說明: 建立 Imputer → StandardScaler → PCA 的安全管線
        2. inputs: n_components, whiten, impute(缺值填補策略)
        3. return: sklearn Pipeline 物件 """
    imputer = SimpleImputer(strategy=impute)
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, whiten=whiten, svd_solver="full", random_state=None)
    return Pipeline([("imputer", imputer), ("scaler", scaler), ("pca", pca)])


def topn_loadings(components: np.ndarray, feature_names: List[str], n: int) -> pd.DataFrame:
    """ 1. 說明: 取每個主成分的前 n 個絕對載荷特徵
        2. inputs: components: PCA.components_ (k, d), feature_names: 原特徵名, n: 每成分輸出數
        3. return: DataFrame（columns: pc, feature, loading, abs_loading） """
    rows = []
    k, _ = components.shape
    for i in range(k):
        load = components[i, :]
        order = np.argsort(np.abs(load))[::-1][:n]
        for j in order:
            rows.append({
                "pc": f"PC{i+1}",
                "feature": feature_names[j],
                "loading": float(load[j]),
                "abs_loading": float(abs(load[j]))
            })
    return pd.DataFrame(rows)


def plot_cumvar(mean_evr: np.ndarray, outpath: Path) -> None:
    """ 1. 說明: 繪製平均累積解釋變異曲線
        2. inputs: mean_evr: 平均的 explained_variance_ratio_（依成分）
                   outpath: 圖檔輸出路徑（.png）
        3. return: None（直接存檔） """
    plt.figure()
    cum = np.cumsum(mean_evr)
    plt.plot(range(1, len(cum) + 1), cum, marker="o")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Mean Cumulative Explained Variance")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# ----------------------------- Main Runner -----------------------------

def run(cfg: PCAConfig) -> None:
    """ 1. 說明: 主流程—載入資料 → 特徵挑選 → Purged K-Fold → 折內 fit/transform → 輸出結果與報表
        2. inputs: cfg: PCAConfig
        3. return: None（於輸出資料夾寫入多個檔案） """
    outdir = ensure_dir(cfg.output_dir)

    # 讀資料
    df = pd.read_csv(cfg.input_csv)
    if cfg.datetime_col not in df.columns:
        raise ValueError(f"datetime 欄位 {cfg.datetime_col} 不存在")
    df[cfg.datetime_col] = pd.to_datetime(df[cfg.datetime_col], utc=True)
    df.sort_values(cfg.datetime_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 欄位選取（自動排除 y_* 與 exclude_patterns）
    feat_cols = select_feature_columns(
        df=df,
        datetime_col=cfg.datetime_col,
        exclude_patterns=cfg.exclude_patterns,
        include_prefixes=None,
        auto_exclude_labels=True,
    )
    if not feat_cols:
        raise ValueError("沒有可用的連續特徵欄位可做 PCA，請檢查 exclude_patterns 設定")

    X = df[feat_cols].copy()
    times = pd.DatetimeIndex(df[cfg.datetime_col])
    splits = build_purged_kfold_indices(times, cfg.n_splits, cfg.embargo_minutes, cfg.shuffle)

    # 逐折訓練與轉換（折內 fit → transform）
    Z_all: list[np.ndarray | None] = [None] * len(df)
    fold_summaries: list[Dict] = []
    evr_matrix: list[list[float]] = []
    all_loadings = []

    for fold_id, (tr_idx, te_idx) in enumerate(splits, start=1):
        pipe = make_pipeline(cfg.n_components, cfg.whiten, cfg.impute)
        pipe.fit(X.iloc[tr_idx, :])

        pca: PCA = pipe.named_steps["pca"]
        k = pca.components_.shape[0]

        idx_fold = np.concatenate([tr_idx, te_idx])
        Z_fold = pipe.transform(X.iloc[idx_fold, :])
        for i_local, i_global in enumerate(idx_fold):
            Z_all[i_global] = Z_fold[i_local, :]

        evr = pca.explained_variance_ratio_
        evr_matrix.append(evr.tolist())

        df_top = topn_loadings(pca.components_, feat_cols, cfg.topn_loadings)
        df_top.insert(0, "fold", fold_id)
        all_loadings.append(df_top)

        fold_summaries.append({
            "fold": fold_id,
            "n_components": int(k),
            "cum_explained_variance": float(np.cumsum(evr)[-1]),
        })

    # 合併結果 → 輸出 CSV：datetime + PC1..PCk
    max_k = max(s["n_components"] for s in fold_summaries)
    Z_mat = np.full((len(df), max_k), np.nan, dtype=float)
    for i, row in enumerate(Z_all):
        if row is None:
            continue
        k_i = min(max_k, len(row))
        Z_mat[i, :k_i] = row[:k_i]

    pc_cols = [f"PC{j+1}" for j in range(max_k)]
    out_df = pd.concat(
        [df[[cfg.datetime_col]].reset_index(drop=True), pd.DataFrame(Z_mat, columns=pc_cols)],
        axis=1,
    )
    out_df.to_csv(outdir / "pca_output.csv", index=False)

    # 輸出彙總
    pd.DataFrame(fold_summaries).to_csv(outdir / "pca_summary.csv", index=False)
    loadings_df = pd.concat(all_loadings, axis=0, ignore_index=True)
    loadings_df.to_csv(outdir / "loadings_topn.csv", index=False)

    # 平均 EVR 與 Scree 圖
    max_len = max(len(e) for e in evr_matrix)
    evr_pad = np.full((len(evr_matrix), max_len), np.nan)
    for r, e in enumerate(evr_matrix):
        evr_pad[r, : len(e)] = np.array(e)
    mean_evr = np.nanmean(evr_pad, axis=0)
    pd.DataFrame({"pc": [f"PC{j+1}" for j in range(len(mean_evr))], "mean_explained_variance_ratio": mean_evr}).to_csv(
        outdir / "mean_evr.csv", index=False
    )
    plot_cumvar(mean_evr, outdir / "cumulative_variance.png")

    # 存 meta.json 以便 reporter 後處理
    meta = {
        "feature_cols": feat_cols,
        "pc_cols": pc_cols,
        "fold_summaries": fold_summaries,
        "evr_matrix": evr_matrix,
        "config": cfg.__dict__,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[PCA] 完成，輸出在：{outdir}")


def parse_args() -> argparse.Namespace:
    """ 1. 說明: 解析命令列參數
        2. inputs: 無（直接讀取 sys.argv）
        3. return: argparse.Namespace """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="pca_config.yaml 路徑")
    return ap.parse_args()


def main() -> None:
    """ 1. 說明: 入口點—讀設定 → 執行 run()
        2. inputs: 無
        3. return: None """
    args = parse_args()
    raw = load_yaml(args.config)
    cfg = PCAConfig(**raw)
    run(cfg)


if __name__ == "__main__":
    main()

"""
使用方式：
python feature_selection/statistics/pca/run_pca.py --config feature_selection/statistics/pca/pca_config.yaml

注意：
1) 請刪除 pca_config.yaml 內的 label_cols。
2) 程式會自動排除所有 y_* 欄位與 exclude_patterns（如 *_flag），輸出不再附帶任何標籤欄位。
3) 需要標籤建模時，請以 datetime 作 key 自行 merge 回來，並保持與 PCA 同一套 Purged K-Fold 設定做 OOS 驗證。
"""

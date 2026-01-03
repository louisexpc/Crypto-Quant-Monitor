# feature_selection/statistics/ts_kmeans_msm/make_msm_grid_csv.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import pandas as pd


@dataclass
class MSMGridCSVFormatter:
    """
    1. 說明:
       讀取 MSM grid 的 summary CSV，統一欄位名稱與格式，選擇需要的欄位，
       並輸出一份適合放進論文表格用的精簡 CSV（數值四捨五入到小數點後三位）。

    2. inputs:
       - in_csv:  原始 summary CSV 路徑。
       - out_csv: 輸出的精簡 CSV 路徑。
       - k_corr:  若指定，則只保留該 k_corr 的列；為 None 則保留全部。

    3. return:
       - 無回傳值；請呼叫 save() 產生輸出檔案。
    """

    in_csv: Path
    out_csv: Path
    k_corr: Optional[int] = None

    def run(self) -> None:
        """
        1. 說明:
           主流程：讀檔 → 正規化欄位 → 過濾/排序 → 四捨五入 → 輸出 CSV。

        2. inputs:
           - 無（使用物件屬性）。

        3. return:
           - 無；直接在 out_csv 寫出結果。
        """
        df = self._load_and_normalize()
        df = self._filter_and_sort(df)
        df = self._round_metrics(df)
        self._save(df)

    def _load_and_normalize(self) -> pd.DataFrame:
        """
        1. 說明:
           讀取原始 CSV，並把常見欄位名稱統一成統一格式。

        2. inputs:
           - 無（使用 self.in_csv）。

        3. return:
           - df: 欄位已統一命名的 DataFrame，至少包含:
                 k_corr, k_msm, silhouette_msm,
                 cluster_intra_mean_avg, cluster_inter_mean_avg,
                 mean_corr_representatives。
        """
        path = Path(self.in_csv)
        if not path.exists():
            raise FileNotFoundError(f"[make_msm_grid_csv] CSV not found: {path}")

        df = pd.read_csv(path)

        # 允許兩種命名: (m,n) 或 (k_corr,k_msm)
        rename_map = {}
        if "m" in df.columns and "k_corr" not in df.columns:
            rename_map["m"] = "k_corr"
        if "n" in df.columns and "k_msm" not in df.columns:
            rename_map["n"] = "k_msm"

        # 允許 intra/inter 平均的不同命名
        if "cluster_intra_mean_avg" not in df.columns:
            if "intra_mean_avg" in df.columns:
                rename_map["intra_mean_avg"] = "cluster_intra_mean_avg"
        if "cluster_inter_mean_avg" not in df.columns:
            if "inter_mean_avg" in df.columns:
                rename_map["inter_mean_avg"] = "cluster_inter_mean_avg"

        if rename_map:
            df = df.rename(columns=rename_map)

        required: List[str] = [
            "k_corr",
            "k_msm",
            "silhouette_msm",
            "cluster_intra_mean_avg",
            "cluster_inter_mean_avg",
            "mean_corr_representatives",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                "[make_msm_grid_csv] CSV 缺少必要欄位: "
                f"{missing}\n目前欄位: {list(df.columns)}"
            )

        # 基本型別處理
        df["k_corr"] = df["k_corr"].astype(int)
        df["k_msm"] = df["k_msm"].astype(int)

        return df

    def _filter_and_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1. 說明:
           依照 k_corr 過濾（若有指定），並按照 k_corr, k_msm 排序。

        2. inputs:
           - df: 前一階段處理好的 DataFrame。

        3. return:
           - new_df: 過濾並排序後的 DataFrame。
        """
        if self.k_corr is not None:
            df = df[df["k_corr"] == self.k_corr].copy()
            if df.empty:
                raise ValueError(
                    f"[make_msm_grid_csv] 沒有找到 k_corr={self.k_corr} 的列，"
                    "請檢查 --kcorr 或輸入 CSV 內容。"
                )

        df = df.sort_values(["k_corr", "k_msm"]).reset_index(drop=True)
        # 只留下要用的欄位
        cols = [
            "k_corr",
            "k_msm",
            "silhouette_msm",
            "cluster_intra_mean_avg",
            "cluster_inter_mean_avg",
            "mean_corr_representatives",
        ]
        return df[cols]

    def _round_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1. 說明:
           對所有連續指標欄位做四捨五入到小數點後第三位。

        2. inputs:
           - df: 完整欄位的 DataFrame。

        3. return:
           - new_df: 指標欄位已四捨五入的 DataFrame。
        """
        metric_cols = [
            "silhouette_msm",
            "cluster_intra_mean_avg",
            "cluster_inter_mean_avg",
            "mean_corr_representatives",
        ]
        df = df.copy()
        for col in metric_cols:
            df[col] = df[col].round(3)
        return df

    def _save(self, df: pd.DataFrame) -> None:
        """
        1. 說明:
           將結果寫出到 out_csv。

        2. inputs:
           - df: 要輸出的 DataFrame。

        3. return:
           - 無。
        """
        out_path = Path(self.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[make_msm_grid_csv] 寫出 {len(df)} 列到 {out_path}")


def main() -> None:
    """
    1. 說明:
       CLI 入口：解析參數並呼叫 MSMGridCSVFormatter。

    2. inputs:
       - 由命令列提供:
         --in_csv:  原始 summary CSV 路徑。
         --out_csv: 輸出的精簡 CSV 路徑。
         --kcorr:   （選填）固定的 k_corr 值。

    3. return:
       - 無。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_csv",
        type=Path,
        required=True,
        help="原始 summary CSV 路徑。",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        required=True,
        help="輸出的精簡 CSV 路徑。",
    )
    parser.add_argument(
        "--kcorr",
        type=int,
        default=None,
        help="若指定，則只輸出該 k_corr 的 rows。",
    )
    args = parser.parse_args()

    formatter = MSMGridCSVFormatter(
        in_csv=args.in_csv,
        out_csv=args.out_csv,
        k_corr=args.kcorr,
    )
    formatter.run()


if __name__ == "__main__":
    main()

"""
python feature_selection/statistics/ts_kmeans_msm/make_msm_grid_table.py \
  --in_csv feature_selection/results/t_sne_grid/summary.csv \
  --out_csv Report/tables/msm_grid_kcorr60.csv \
  --kcorr 60

python feature_selection/statistics/ts_kmeans_msm/make_msm_grid_table.py \
  --in_csv feature_selection/results/t_sne_grid/summary.csv \
  --out_csv Report/tables/msm_grid_full.csv


"""
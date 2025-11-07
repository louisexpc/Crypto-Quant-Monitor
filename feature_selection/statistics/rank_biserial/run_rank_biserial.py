# feature_selection/statistics/rank_biserial/run_rank_biserial.py
from __future__ import annotations
from typing import Dict, Tuple
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# 你現成的時間序列 CV
from train.data.folds import FoldGenerator, split_fold_to_indices

# 單變量評分器
from feature_selection.statistics.rank_biserial.scorer import VetScorer

class EventData:
    """
    1. 說明:
        讀入 TBM_label.csv 與 precomputed.csv，統一轉成 UTC tz-aware。
        依設定將「過去 N 根」彙整成單一標量（例如 ema/zscore/slope 等），
        再把各事件 t0 對齊到「t0 之前」最近一筆（可選嚴格 <t0 與最大回溯距離）。
    2. inputs:
        cfg (dict): 由 YAML 讀入的設定。
    3. return:
        - evt_df: DataFrame(index=t0[UTC]; cols=['y','side','entry_price','t1'])
        - feat_at_t0: DataFrame(index=t0; 各特徵在事件時刻的彙整值)
        - bars_feat: DataFrame(bar×特徵，用於切 fold 的時間座標)
    """
    def __init__(self, cfg: Dict):
        self.cfg = cfg

    @staticmethod
    def _ensure_utc(ts: pd.Series | pd.DatetimeIndex, tz_in: str) -> pd.DatetimeIndex:
        """
        1. 說明: 將輸入時間轉為 UTC tz-aware。
        2. inputs:
            ts: Series/DatetimeIndex
            tz_in: 原本時區（如 'Asia/Taipei' 或 'UTC'）
        3. return: DatetimeIndex (UTC tz-aware)
        """
        idx = pd.DatetimeIndex(ts)
        if idx.tz is None:
            idx = idx.tz_localize(tz_in)
        else:
            idx = idx.tz_convert(tz_in)
        return idx.tz_convert("UTC")

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        paths = self.cfg["paths"]; idx_cfg = self.cfg["index"]

        tbm = pd.read_csv(paths["tbm_labels"])
        bars = pd.read_csv(paths["precomputed"])

        # 時區處理
        bars_index = self._ensure_utc(bars[idx_cfg["dt_col_precomputed"]], idx_cfg["tz_precomputed"])
        bars.index = bars_index
        bars.index.name = "dt"
        tbm["t0"] = self._ensure_utc(tbm["t0"], self.cfg["index"]["tz_labels"])
        tbm["t1"] = self._ensure_utc(tbm["t1"], self.cfg["index"]["tz_labels"])
        tbm = tbm.set_index("t0").sort_index()

        # y ∈ {0,1}
        if "label" not in tbm.columns:
            raise ValueError("TBM_label.csv 缺少 'label' 欄位")
        y = (tbm["label"] > 0).astype(int)

        # ---- 健壯 side 正規化：支援 1/-1、"1"/"-1"、"long"/"short"、"buy"/"sell" ----
        side_raw = tbm.get("side", pd.Series(index=tbm.index, dtype="object"))
        side_num = pd.to_numeric(side_raw, errors="coerce")

        if side_num.isna().any():
            # 試圖由字串語彙對映
            side_map = {
                "1": 1, "-1": -1, "long": 1, "short": -1, "buy": 1, "sell": -1,
                "多": 1, "空": -1
            }
            side_num = (
                side_raw.astype(str)
                .str.strip().str.lower()
                .map(side_map)
                .astype("float64")
            )

        # 仍有缺失就當缺資料
        if side_num.isna().any():
            # 保留 NaN，後續若有 side 過濾會自然被剔除
            pass

        # 將非 {±1} 的數值（若有）以符號收斂為 ±1；0 仍視為 NaN
        side_num = side_num.where(side_num.isin([1.0, -1.0]), np.sign(side_num))
        side_num = side_num.where(side_num != 0, np.nan)

        evt_df = pd.DataFrame({
            "y": y.values,
            "side": side_num.values,
            "entry_price": tbm.get("entry_price", np.nan).values,
            "t1": tbm["t1"].values,
        }, index=tbm.index)

        # 依 side 過濾（all/long/short）
        choice = str(self.cfg.get("filter", {}).get("side", "all")).lower()
        if choice not in ("all", "long", "short"):
            raise ValueError(f"[filter.side] 僅支援 all/long/short，收到: {choice}")
        if choice != "all":
            target = 1 if choice == "long" else -1
            evt_df = evt_df[evt_df["side"] == float(target)]
            if evt_df.empty:
                raise ValueError(f"[filter.side={choice}] 篩到 0 筆事件，請檢查 TBM_label.csv 的 side 欄位內容與格式")

        # 特徵欄位過濾
        col_cfg = self.cfg.get("columns", {})
        exclude_exact = set(col_cfg.get("exclude_exact", [
            "datetime","timestamp","open","high","low","close","volume"
        ]))
        exclude_prefix = list(col_cfg.get("exclude_prefix", []))

        cols = [c for c in bars.columns if c not in exclude_exact]
        for p in exclude_prefix:
            cols = [c for c in cols if not c.startswith(p)]
        if len(cols) == 0:
            raise ValueError("[columns] 過濾後無可用特徵欄位，請檢查 exclude_exact / exclude_prefix 設定")

        bars_feat = bars[cols].sort_index()

        # 合成版：先做窗口彙整，再對齊事件 t0（支援嚴格上一根與最大容忍落後）
        feat_at_t0 = self._lookback(evt_df.index, bars_feat)

        return evt_df, feat_at_t0, bars_feat

    def _lookback(
        self,
        event_times: pd.DatetimeIndex,
        bars_feat: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        1. 說明:
            在 bar 級對每個特徵做「窗口彙整」→ 再把事件 t0 對齊到彙整後序列的
            「t0 之前最近一筆」取值，得到事件時刻的單一分數。
            支援的彙整方法：last / sma / ema / zscore / ewm_zscore / slope / percentile
            並提供:
            - strictly_previous: True 時嚴格取 < t0（不含等於）
            - max_lag: 允許的最大回溯距離（超過則回傳 NaN）
        2. inputs:
            event_times (DatetimeIndex): 事件 t0（UTC tz-aware）
            bars_feat (DataFrame): bar×特徵，index=UTC（index.name 可為任意）
        3. return:
            DataFrame: index=event_times；欄=特徵；值=各事件的彙整後分數
        """
        ev = self.cfg.get("events", {})
        method = str(ev.get("reduce", "last")).lower()
        W = int(ev.get("window_bars", 36))
        hl = ev.get("halflife_bars", max(2, W // 2))
        q = float(ev.get("percentile_q", 0.8))
        strictly_previous = bool(ev.get("strictly_previous", False))
        max_lag = ev.get("max_lag", None)  # 例如 "2h"、"1D"；None 表不限制

        fb = bars_feat.sort_index().copy()
        if fb.index.tz is None:
            fb.index = fb.index.tz_localize("UTC")
        else:
            fb.index = fb.index.tz_convert("UTC")
        fb.index.name = "dt"

        # ---- 窗口彙整（先把 bar→bar 摘要）----
        if method == "last":
            reduced = fb
        else:
            minp = max(3, W // 4)

            if method == "sma":
                reduced = fb.rolling(W, min_periods=minp).mean()
            elif method == "ema":
                reduced = fb.ewm(halflife=hl, adjust=False, min_periods=minp).mean()
            elif method == "zscore":
                m = fb.rolling(W, min_periods=minp).mean()
                s = fb.rolling(W, min_periods=minp).std(ddof=0)
                reduced = (fb - m) / (s + 1e-12)
            elif method == "ewm_zscore":
                m = fb.ewm(halflife=hl, adjust=False, min_periods=minp).mean()
                s = fb.ewm(halflife=hl, adjust=False, min_periods=minp).std(bias=False)
                reduced = (fb - m) / (s + 1e-12)
            elif method == "slope":
                import numpy as np
                def _slope(v: np.ndarray) -> float:
                    x = np.arange(v.shape[0], dtype=float)
                    return float(np.polyfit(x, v, 1)[0]) if np.isfinite(v).all() else np.nan
                reduced = fb.rolling(W, min_periods=max(5, W // 3)).apply(_slope, raw=True)
            elif method == "percentile":
                reduced = fb.rolling(W, min_periods=minp).quantile(q)
            else:
                raise ValueError(f"[events.reduce] 不支援的方法: {method}")

        # ---- 事件對齊：取 t0 之前最近一筆（可選嚴格 <t0 與最大回溯距離）----
        left = pd.DataFrame({"dt": pd.DatetimeIndex(event_times)}).sort_values("dt")
        if left["dt"].dt.tz is None:
            left["dt"] = left["dt"].dt.tz_localize("UTC")
        right = reduced.reset_index().sort_values("dt")

        merge_kwargs = dict(
            left=left, right=right, on="dt", direction="backward",
            allow_exact_matches=not strictly_previous,
        )
        if max_lag is not None:
            merge_kwargs["tolerance"] = pd.Timedelta(max_lag)

        out = pd.merge_asof(**merge_kwargs).set_index("dt")
        return out.reindex(event_times)  # 保持事件原順序

def _diagnose(bars: pd.DataFrame, evt_df: pd.DataFrame, folds, feat_at_t0: pd.DataFrame):
    """
    1. 說明:
        輸出基本診斷資訊：bars/事件時間範圍、各折測試區間事件數、非NaN佔比最高的特徵。
    2. inputs:
        bars: bar×特徵 DataFrame
        evt_df: 事件 DataFrame（index=t0）
        folds: 折疊清單
        feat_at_t0: 事件×特徵（事件時刻值）
    3. return:
        無（標準輸出印資訊）
    """
    print(f"[debug] events after side filter: {len(evt_df)}")
    print(f"[debug] bars range (UTC):   {bars.index.min()} ~ {bars.index.max()}")
    print(f"[debug] events range (UTC): {evt_df.index.min()} ~ {evt_df.index.max()}")
    for i, fold in enumerate(folds):
        te = pd.DatetimeIndex(fold["test_times"])
        te_n = ((evt_df.index >= te.min()) & (evt_df.index <= te.max())).sum()
        print(f"[debug] fold#{i+1} test_month={fold.get('test_month')}"
              f" | bars: {te.min()} ~ {te.max()} | events_in_test={te_n}")
    nnr = (~feat_at_t0.isna()).mean().sort_values(ascending=False).head(10)
    print("[debug] top non-NaN ratio features:\n", nnr)


def _export_selected_features(cfg: Dict, selected_cols: list[str]) -> None:
    """
    1. 說明:
        讀原始 precomputed CSV，取 selected_cols 的交集並輸出到指定路徑。
        同時輸出欄名清單（若設定）。
    """
    if not selected_cols:
        print("[feature_select] no rank-biserial selected features to export.")
        return

    paths = cfg.get("paths", {})
    source_path = Path(paths["precomputed"])
    if not source_path.exists():
        print(f"[WARN] precomputed source not found: {source_path}")
        return

    df = pd.read_csv(source_path)
    idx_cfg = cfg.get("index", {})
    base_cols: list[str] = []
    dt_col = idx_cfg.get("dt_col_precomputed")
    for col in [dt_col, "datetime", "timestamp"]:
        if col and col in df.columns and col not in base_cols:
            base_cols.append(col)

    keep_cols = [col for col in selected_cols if col in df.columns]
    missing = sorted(set(selected_cols) - set(keep_cols))
    if missing:
        print(f"[WARN] missing {len(missing)} selected features in source CSV; skipped.")

    if not keep_cols:
        print("[feature_select] selected features not present in source CSV, nothing exported.")
        return

    selected_df = df[base_cols + keep_cols] if base_cols else df[keep_cols]

    out_csv = paths.get("selected_feat_csv")
    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        selected_df.to_csv(out_path, index=False)
        print(f"[feature_select] selected feature matrix saved → {out_path}")

    cols_txt = paths.get("selected_feat_cols")
    if cols_txt:
        txt_path = Path(cols_txt)
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text("\n".join(keep_cols), encoding="utf-8")
        print(f"[feature_select] selected feature list saved → {txt_path}")


def main():
    """
    1. 說明:
        讀 YAML → 載入事件與特徵 → 用 train.data.folds 產生時間折疊
        → 計算各特徵 OOS AUC 與 r_rb → U 檢定 + BH-FDR → 匯出 CSV。
    2. inputs:
        --config: 路徑，預設 feature_select/config.yaml
        --side  : 覆蓋 YAML 的 filter.side（all|long|short）
        --debug : 印診斷資訊
    3. return:
        於 cfg.paths.out_csv 輸出彙整表；stdout 列印前 20 筆。
    """
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--side", type=str, choices=["all", "long", "short"], default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg: Dict = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.side is not None:
        cfg.setdefault("filter", {})["side"] = args.side

    # 載資料
    loader = EventData(cfg)
    evt_df, feat_at_t0, bars = loader.load()

    # 產生 folds（沿用你的 train/data/folds.py）
    fg = FoldGenerator(
        dt_index=bars.index,
        mode=cfg["cv"]["mode"],
        start_month=cfg["cv"].get("start_date"),
        end_month=cfg["cv"].get("end_date"),
    )
    folds = (
        fg.make_purged_kfold(
            n_splits=int(cfg["cv"]["n_splits"]),
            embargo_hours=int(cfg["cv"]["embargo_hours"]),
            min_train_days=30,
        )
        if cfg["cv"]["mode"] == "purged_kfold"
        else fg.make_rolling_folds(train_window=6, embargo_hours=int(cfg["cv"]["embargo_hours"]), test_freq="M")
    )

    if args.debug:
        _diagnose(bars, evt_df, folds, feat_at_t0)

    # 計分
    scorer = VetScorer(cfg, evt_df, feat_at_t0, bars)
    res = scorer.run(folds)

    # 若為空，給出可操作的提示（不拋錯，以利自動化批次）
    if res is None or len(res) == 0:
        print("[feature_select] 結果為空。建議檢查：")
        print("  1) config.index.tz_precomputed 是否應為 'Asia/Taipei'（而非 'UTC'）？")
        print("  2) config.scoring.min_non_nan 是否過大（先降到 20～50 試跑）？")
        print("  3) config.columns.exclude_prefix 是否把主要特徵（如 'm_-' 前綴）排光？")
        print("  4) 折疊設定 n_splits / test_freq 是否讓測試月沒有事件？（試改 rolling 或減少 n_splits）")

    Path(cfg["paths"]["out_csv"]).parent.mkdir(parents=True, exist_ok=True)

    out_df = res if res is not None else pd.DataFrame()
    # 1) 存檔時四位小數
    out_df.to_csv(cfg["paths"]["out_csv"], float_format="%.4f")

    print(f"[feature_select] saved → {cfg['paths']['out_csv']}")
    # 2) 終端列印時四位小數
    if len(out_df):
        print(out_df.head(20).round(4))

        if "fdr_reject" in out_df.columns:
            mask = out_df["fdr_reject"].astype(bool)
            selected_cols = mask[mask].index.tolist()
        else:
            selected_cols = []

        top_k = int(cfg.get("scoring", {}).get("top_k", 0))
        if top_k > 0:
            additional = out_df.index[:top_k].tolist()
            selected_cols = list(dict.fromkeys(selected_cols + additional))

        _export_selected_features(cfg, selected_cols)
    else:
        _export_selected_features(cfg, [])

if __name__ == "__main__":
    main()

"""
用法：
python feature_selection/statistics/rank_biserial/run_rank_biserial.py --side long --debug

# 如 precomputed 的時間其實是台北本地時間，請在 YAML 設定：
# index.tz_precomputed: "Asia/Taipei"
"""

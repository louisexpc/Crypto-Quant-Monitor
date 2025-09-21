# dataloader.py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Literal, Optional, List, Dict

from .scalar import pick_cols_to_scale, _get_scaler, ColumnSubsetScaler
from .build_features import create_label

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.time_dataset import SeqDataset
from dataset.event_dataset import EventDataset

def split_fold_to_indices(df: pd.DataFrame, fold: Dict, cfg: Dict):
    """
    1. 說明:
        根據 fold 切出 train/val/test 的索引：
        - 若 fold 的布林遮罩長度與 df 相同，直接使用布林遮罩（time-driven 情境）。
        - 否則改用 fold['train_val_times'] / fold['test_times'] 做「時間對齊」（event-driven 推薦）。
        - 回傳三段 DatetimeIndex（已按時間排序）。

    2. inputs:
        df (DataFrame): 目標資料（可能是 bar 級，或事件 t0 級）
        fold (dict):     FoldGenerator 產生的折疊資訊
        cfg (dict):      含 train/val split 比例（cfg['cv']['train_val_split']）

    3. return:
        tr_idx, va_idx, te_idx (DatetimeIndex, DatetimeIndex, DatetimeIndex)
    """
    train_val_mask = fold.get("train_val_mask")
    test_mask = fold.get("test_mask")

    # 標準化 df.index → UTC tz-aware
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")

    # 判斷是否可直接用布林遮罩
    use_mask_direct = (
        isinstance(train_val_mask, (np.ndarray, pd.Series)) and
        isinstance(test_mask, (np.ndarray, pd.Series)) and
        len(train_val_mask) == len(df) and
        len(test_mask) == len(df)
    )

    if use_mask_direct:
        df_tv = df.loc[np.asarray(train_val_mask).astype(bool)].sort_index()
        df_te = df.loc[np.asarray(test_mask).astype(bool)].sort_index()
    else:
        # 後備：用時間集合對齊（FoldGenerator 需在 folds.append 時提供 times）
        tv_times = fold.get("train_val_times")
        te_times = fold.get("test_times")
        if tv_times is None or te_times is None:
            raise ValueError(
                "Fold boolean mask 與 df 長度不一致，且缺少 'train_val_times' / 'test_times' 可供時間對齊。"
                "請更新 FoldGenerator 在每折輸出時間集合。"
            )

        tv_times = pd.DatetimeIndex(tv_times)
        te_times = pd.DatetimeIndex(te_times)
        tv_times = tv_times.tz_localize("UTC") if tv_times.tz is None else tv_times.tz_convert("UTC")
        te_times = te_times.tz_localize("UTC") if te_times.tz is None else te_times.tz_convert("UTC")

        tv_mask_local = idx.isin(tv_times)
        te_mask_local = idx.isin(te_times)

        df_tv = df.loc[tv_mask_local].sort_index()
        df_te = df.loc[te_mask_local].sort_index()

    # train/val split（時間順序）
    split_ratio = float(cfg["cv"]["train_val_split"])
    split_idx = int(len(df_tv) * split_ratio)

    tr_idx = df_tv.index[:split_idx]
    va_idx = df_tv.index[split_idx:]
    te_idx = df_te.index

    return tr_idx, va_idx, te_idx


def select_plan_columns(feat_df: pd.DataFrame, cfg: Dict) -> List[str]:
    """
    根據 cfg.features.plan 產生要使用的欄位集合，包含：
    - OHLCV（僅在 plan 中對應項目 enabled 時才保留）
    - 所有 1-min 欄位（以 DEFAULT_MINUTE_PREFIXES 為前綴，例如 'm_...'; 請確保已於上游 drop 掉 datetime/timestamp）
    - 由 plan 中 enabled 的特徵映射出的實際欄位（與 feat_df.columns 取交集）

    輸出順序遵守原始 feat_df.columns 的順序。
    """
    from .indicators import FeatureComputer, DEFAULT_MINUTE_PREFIXES
    import re

    cols_all = list(map(str, feat_df.columns))
    # OHLCV 按 plan 控制是否納入特徵（label 計算已在外層處理，不受此處影響）
    ohlcv_keep: set[str] = set()

    # Minute (1-min flattened) selection via config.features.min_trade_feat
    # Expect column form like: m_[-lag]_<base>
    min_feat_list = list(((cfg.get("features", {}) or {}).get("min_trade_feat", [])) or [])
    minute_cols: set[str] = set()
    if min_feat_list:
        m_pat = re.compile(r"^m_(-?\d+)_(.+)$")
        for c in cols_all:
            if not any(str(c).startswith(p) for p in DEFAULT_MINUTE_PREFIXES):
                continue
            m = m_pat.match(str(c))
            if not m:
                continue
            base = m.group(2)
            if base in min_feat_list:
                minute_cols.add(c)

    plan = (cfg.get("features", {}) or {}).get("plan", {}) or {}
    try:
        specs = FeatureComputer._enabled_features(plan)
    except Exception:
        specs = []

    want = set()

    def add_if_present(names):
        if isinstance(names, str):
            if names in feat_df.columns:
                want.add(names)
        else:
            for n in names:
                if n in feat_df.columns:
                    want.add(n)

    for item in specs:
        name = str(item.get("name", "")).upper()
        kw = item.get("kwargs", {}) or {}

        if name == "SMA":
            L = int(kw.get("length", 0)); s = str(kw.get("sign", "false")).lower()
            if s in ("true","sign","both"): add_if_present(f"SSMA_{L}")
            if s in ("false","cont","both"): add_if_present(f"SMA_{L}")
        elif name == "EMA":
            L = int(kw.get("length", 0)); s = str(kw.get("sign", "false")).lower()
            if s in ("true","sign","both"): add_if_present(f"SEMA_{L}")
            if s in ("false","cont","both"): add_if_present(f"EMA_{L}")
        elif name == "TEMA":
            L = int(kw.get("length", 0)); s = str(kw.get("sign", "false")).lower()
            if s in ("true","sign","both"): add_if_present(f"STEMA_{L}")
            if s in ("false","cont","both"): add_if_present(f"TEMA_{L}")
        elif name == "MACD":
            f = int(kw.get("fast", 12)); sl = int(kw.get("slow", 26)); sg = int(kw.get("signal", 9))
            s = str(kw.get("sign", "false")).lower()
            if s in ("true","sign","both"): add_if_present(f"SMACD_{f}_{sl}_{sg}")
            if s in ("false","cont","both"): add_if_present(f"MACD_{f}_{sl}_{sg}")
        elif name == "SLOPE":
            L = int(kw.get("length", 0)); add_if_present(f"SLOPE_{L}")
        elif name == "TTM_TRND":
            L = int(kw.get("length", 6)); add_if_present(f"TTM_TRND_{L}")
        elif name == "DPO":
            L = int(kw.get("length", 0)); add_if_present(f"DPO_{L}")
        elif name == "AMATE_LR":
            f = int(kw.get("fast", 8)); sl = int(kw.get("slow", 21)); m = int(kw.get("mamode", 2))
            add_if_present(f"AMATe_LR_{f}_{sl}_{m}")

        elif name == "RSI":
            L = int(kw.get("length", 14)); add_if_present(f"RSI_{L}")
        elif name == "MOM":
            L = int(kw.get("length", 30)); add_if_present(f"MOM_{L}")
        elif name == "STOCH":
            k = int(kw.get("k", 14)); add_if_present([f"STOCHk_{k}", f"STOCHd_{k}"])
        elif name == "KDJ":
            k = int(kw.get("k", 9)); d = int(kw.get("d", 3)); add_if_present(f"J_{k}_{d}")
        elif name == "UO":
            f = int(kw.get("fast", 7)); md = int(kw.get("medium", 14)); sl = int(kw.get("slow", 28)); add_if_present(f"UO_{f}_{md}_{sl}")
        elif name == "RVI":
            L = int(kw.get("length", 14)); add_if_present(f"RVI_{L}")
        elif name == "CCI":
            L = int(kw.get("length", 14)); c = float(kw.get("c", 0.015)); add_if_present(f"CCI_{L}_{c}")
        elif name == "ZS":
            L = int(kw.get("length", 30)); add_if_present(f"ZS_{L}")
        elif name == "WILLR":
            L = int(kw.get("length", 14)); add_if_present(f"WILLR_{L}")

        elif name == "TRUERANGE":
            add_if_present("TRUERANGE_1")
        elif name == "RANGE":
            W = int(kw.get("window", 24)); add_if_present(f"RANGE_{W}")
        elif name == "ATR":
            L = int(kw.get("length", 14)); add_if_present(f"ATR_{L}")
            if bool(kw.get("pct", True)): add_if_present(f"ATRP_{L}")
        elif name == "MASSI":
            f = int(kw.get("fast", 9)); sl = int(kw.get("slow", 25)); add_if_present(f"MASSI_{f}_{sl}")
        elif name == "BBP":
            L = int(kw.get("length", 5)); st = float(kw.get("std", 2.0)); add_if_present(f"BBP_{L}_{st}")
        elif name == "EWMRET":
            hls = kw.get("halflife", [])
            if isinstance(hls, int): hls = [hls]
            for hl in hls:
                add_if_present([f"EWM_M_{int(hl)}", f"EWM_S_{int(hl)}"])

        elif name == "PVO":
            pv_cols = [c for c in cols_all if str(c).startswith("PVO_") or c == "PVO"]
            add_if_present(pv_cols)
        elif name == "PVR":
            add_if_present("PVR")
        elif name == "BOP":
            add_if_present("BOP")
        elif name == "PXVOL":
            add_if_present(["DIR_STRENGTH","PXV_LR_VCHG","DIRxVOL"])

        elif name == "LOGRET":
            lags = kw.get("lags", [])
            lags = lags if isinstance(lags, (list, tuple)) else [lags]
            for k in lags:
                add_if_present(f"LOGRET_{int(k)}")
        elif name == "TIME_CYC":
            if bool(kw.get("daily", True)):
                add_if_present(["TOD_SIN","TOD_COS"])
            if bool(kw.get("weekly", True)):
                add_if_present(["DOW_SIN","DOW_COS"])

        elif name == "FOUND":
            add_if_present("funding_rate")
        elif name == "M15_DIR":
            add_if_present(["M15_DIR_01","M15_DIR_12","M15_DIR_23"])
        elif name == "M15_VOL":
            add_if_present(["M15_VOL_0","M15_VOL_1","M15_VOL_2","M15_VOL_3"])
        elif name == "FNG_IDX":
            add_if_present(["sent_fng","sent_fng_diff1","sent_fng_z7d"])
        elif name in {"OPEN","HIGH","LOW","CLOSE","VOLUME"}:
            # 僅在 plan 中被 enable 時才保留對應 OHLCV 欄位
            base = name.lower()
            if base in feat_df.columns:
                ohlcv_keep.add(base)
        else:
            # 未知名稱：忽略
            pass

    keep_set = set().union(ohlcv_keep).union(minute_cols).union(want)
    feat_cols = [c for c in cols_all if c in keep_set]
    return feat_cols

# ========== Fold Generator ==========
class FoldGenerator:
    def __init__(self,
                 dt_index: pd.DatetimeIndex,
                 mode: str = "rolling",
                 start_month: str | None = None,
                 end_month: str | None = None,
                 **kwargs):
        """
        1. 說明:
            僅根據 index 生成 folds，支援以 config.yaml 的 cv.start_date / cv.end_date
            限定可用的時間範圍（含起訖），並同時維持 PeriodIndex 與 DatetimeIndex 的時區一致性。
        2. inputs:
            dt_index (DatetimeIndex): 原始時間索引（可 naive 或 tz-aware）。最終統一為 UTC tz-aware。
            mode (str): 目前保留原介面。
            start_month (str|None): 例如 "2023-01-01"；若為 None 則以資料最小時間。
            end_month   (str|None): 例如 "2025-04-30"；若為 None 則以資料最大時間。
        3. return:
            建立好 self.months（以 Period[M] 表示的月份清單）、以及 aware/naive 的起訖邊界。
        """
        # 1) dt_index → UTC tz-aware
        if getattr(dt_index, "tz", None) is None:
            dt_index = dt_index.tz_localize("UTC")
        else:
            dt_index = dt_index.tz_convert("UTC")
        self.dt_index = dt_index
        self.mode = mode
        self.kwargs = kwargs

        # 2) 讀取起訖（naive 與 aware 版本各存一份）
        self.start_ts_naive = pd.Timestamp(start_month) if start_month is not None else pd.Timestamp(dt_index.min().tz_convert("UTC").tz_localize(None))
        self.end_ts_naive   = pd.Timestamp(end_month)   if end_month   is not None else pd.Timestamp(dt_index.max().tz_convert("UTC").tz_localize(None))

        # 對齊到「日期存在」且 start<=end
        if self.end_ts_naive < self.start_ts_naive:
            raise ValueError(f"[FoldGenerator] end_month({self.end_ts_naive}) 早於 start_month({self.start_ts_naive})")

        # aware（UTC）
        self.start_ts_aware = self.start_ts_naive.tz_localize("UTC")
        # end 設為當日 23:59:59.999999999（含訖），避免月底被排除
        self.end_ts_aware   = (self.end_ts_naive + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)).tz_localize("UTC")

        # 3) 先將 dt_index 限縮到 [start,end]（保證後續 months 與遮罩一致）
        in_range = (self.dt_index >= self.start_ts_aware) & (self.dt_index <= self.end_ts_aware)
        self.dt_index = self.dt_index[in_range]
        if len(self.dt_index) == 0:
            raise ValueError("[FoldGenerator] 篩選後的 dt_index 為空，請檢查 cv.start_date / cv.end_date")

        # 4) 建立月份清單（用 naive；PeriodIndex 皆為 naive）
        dt_index_naive = self.dt_index.tz_convert("UTC").tz_localize(None)
        months_all = pd.PeriodIndex(dt_index_naive.to_period("M")).unique().sort_values()
        # 僅保留起訖內的月份（用月份的 start_time 與 naive 起訖比較）
        self.months = [m for m in months_all if (m.start_time >= self.start_ts_naive) and (m.start_time <= self.end_ts_naive)]

    def _get_test_months(self, test_freq: str):
        """
        1. 說明:
            根據 test_freq（'M','Q' 或 '2M','3M' 這類步長）產生要測試的月份，
            並且限制在 [start_date, end_date] 內。
        2. inputs:
            test_freq (str): 測試頻率。
        3. return:
            PeriodIndex 或 list[Period]：測試月份序列。
        """
        if not self.months:
            return []
        # 將起點對齊到月份開頭（避免提供月中日期）
        start_align = pd.Timestamp(self.start_ts_naive.strftime("%Y-%m-01"))
        end_align   = self.months[-1]  # 已根據 end 內縮
        if test_freq in {"M", "Q"}:
            return pd.period_range(start_align, end_align, freq=test_freq)
        else:
            # "2M"/"3M" 等：從已內縮的 months 中以步長取樣
            step = int(test_freq.replace("M", ""))  # "2M" → 2
            return self.months[::step]

    # =========================
    # Rolling folds（原行為維持；僅用內縮後的 months 與 aware 邊界）
    # =========================
    def make_rolling_folds(self, train_window, embargo_hours, test_freq="M"):
        test_months = self._get_test_months(test_freq)
        folds = []
        month_periods = pd.PeriodIndex(self.dt_index.tz_convert("UTC").tz_localize(None).to_period("M"))
        for m in test_months:
            i = self.months.index(m)
            if i < train_window or i + 1 >= len(self.months):
                continue

            train_start = pd.Timestamp(self.months[i - train_window].start_time, tz="UTC")
            test_start  = pd.Timestamp(m.start_time, tz="UTC")
            embargo_delta = pd.Timedelta(hours=embargo_hours)
            train_end   = test_start - embargo_delta

            # 也限制在 end_ts_aware 以內
            train_mask = (self.dt_index >= max(train_start, self.start_ts_aware)) & \
                         (self.dt_index <  min(train_end,   self.end_ts_aware))
            test_mask  = (month_periods == m) & (self.dt_index <= self.end_ts_aware)

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
   
            folds.append({
                'train_val_mask': train_mask,
                'test_mask':      test_mask,
                'test_month':     str(m),
                # 新增：時間集合（對應 bar 級 self.dt_index）
                'train_val_times': self.dt_index[train_mask],
                'test_times':      self.dt_index[test_mask],
            })
        return folds

    # =========================
    # Purged K-fold（含 embargo；同時套用起訖）
    # =========================
    def make_purged_kfold(self,
                          n_splits: int = 5,
                          embargo_hours: int = 24,
                          min_train_days: int = 30) -> list[dict]:
        """
        1. 說明:
            以「時間」切分的 Purged K-Fold（含 embargo）。
            - self.months 先依 cv.start_date / cv.end_date 內縮；
            - 將 months 均分為 n_splits 區段，取每段最後一月為測試月（如 10 個月/5 折 → 2,4,6,8,10）。
            - 訓練集 = 全資料中，剔除 [test_start - embargo, test_end + embargo) 的樣本，
              並限制在 [start_date, end_date] 範圍內。
        2. inputs:
            n_splits (int): 折數（>=2）
            embargo_hours (int): 禁運期（小時）
            min_train_days (int): 訓練天數下限
        3. return:
            list[dict]: 每折 {'train_val_mask','test_mask','test_month'}
        """
        assert n_splits >= 2, "[make_purged_kfold] n_splits 需 >= 2"

        dt_naive = self.dt_index.tz_convert("UTC").tz_localize(None)
        month_periods = pd.PeriodIndex(dt_naive.to_period("M"))
        embargo = pd.Timedelta(hours=int(embargo_hours))

        m_idx = np.arange(len(self.months))
        groups = np.array_split(m_idx, n_splits)

        folds: list[dict] = []
        for g in groups:
            if len(g) == 0:
                continue
            test_m = self.months[g[-1]]
            test_mask = (month_periods == test_m) & (self.dt_index <= self.end_ts_aware)
            if test_mask.sum() == 0:
                continue

            test_start = pd.Timestamp(test_m.start_time, tz="UTC")
            test_end   = pd.Timestamp((test_m + 1).start_time, tz="UTC")
            purge_start, purge_end = test_start - embargo, test_end + embargo

            # 限制在 [start_aware, end_aware] & Purge
            base = (self.dt_index >= self.start_ts_aware) & (self.dt_index <= self.end_ts_aware)
            no_overlap = (self.dt_index < purge_start) | (self.dt_index >= purge_end)
            train_mask = base & no_overlap & (~test_mask)

            if train_mask.sum() == 0:
                continue
            tr_days = (pd.DatetimeIndex(self.dt_index[train_mask]).date[-1] -
                       pd.DatetimeIndex(self.dt_index[train_mask]).date[0]).days + 1
            if tr_days < int(min_train_days):
                continue

            folds.append({
                'train_val_mask': train_mask,
                'test_mask':      test_mask,
                'test_month':     str(test_m),
                # 新增：時間集合（對應 bar 級 self.dt_index）
                'train_val_times': self.dt_index[train_mask],
                'test_times':      self.dt_index[test_mask],
            })
        return folds

# -------------------------
# DataLoader 組裝
# -------------------------

def make_time_loaders_for_fold(df,
                               feat_cols: Optional[List[str]] = None,
                               target_col: Optional[str] = None,
                               fold: Dict = None,
                               cfg: Dict = None,
                               also_XGB: bool = False,
                               pre_feat_df: pd.DataFrame | None = None):
    """
    時間驅動（time-driven）資料載入器（precomputed-only）：
    - 僅從 cfg.features.precomputed.path 載入特徵；不做 runtime 特徵計算。
    - 以預算檔中的 OHLCV 產生 label（create_label）。
    - 依 fold 切出 train/val/test，執行縮放與清理，最後包成三個 DataLoader。
    - TimeSafeScaler：transform_full；sklearn 縮放器：fit on train，再 transform 其他 split。
    """

    task_type = cfg["task"]["type"]
    # 參考折疊的原始索引（由 objective.make_folds 基於此索引生成布林遮罩）
    ref_index = pd.DatetimeIndex(df.index)
    # 僅使用 precomputed 特徵
    pre_path = cfg["data"]["path"]
    if not pre_path and pre_feat_df is None:
        raise ValueError("請在 config.features.precomputed.path 指定預先計算的特徵檔 (.csv 或 .parquet)")
    if pre_feat_df is not None:
        feat_df = pre_feat_df.copy()
    else:
        p = str(pre_path)
        if p.endswith(".csv"):
            feat_df = pd.read_csv(p)
        elif p.endswith(".parquet"):
            feat_df = pd.read_parquet(p)
        else:
            raise ValueError("features.precomputed.path 只支援 .csv 或 .parquet")
        # set index from datetime/timestamp if present
        if "datetime" in feat_df.columns:
            idx = pd.to_datetime(feat_df["datetime"], errors="coerce", utc=True)
            feat_df = feat_df.drop(columns=["datetime"]) 
            feat_df.index = idx
        elif "timestamp" in feat_df.columns:
            ts = pd.to_numeric(feat_df["timestamp"], errors="coerce").astype("Int64")
            unit = "ms" if (ts.dropna().iloc[0] if len(ts.dropna()) else 0) > 1_000_000_000_000 else "s"
            idx = pd.to_datetime(ts, unit=unit, utc=True)
            feat_df = feat_df.drop(columns=["timestamp"]) 
            feat_df.index = idx
        feat_df = feat_df.sort_index()
        feat_df = feat_df[~feat_df.index.duplicated(keep="last")]

        # 對齊時間網格並生成 label（保留 y_reg/y_cls 欄名）
        # 對齊時間網格：以 precomputed 索引生成完整網格
        full_idx = pd.date_range(feat_df.index.min(), feat_df.index.max(), freq=str(cfg["data"]["freq"]), tz="UTC")
        # 檢查預算檔是否包含 OHLCV 欄位
        ohlcv_cols = [c for c in ["open","high","low","close","volume"] if c in feat_df.columns]
        if len(ohlcv_cols) < 5:
            raise KeyError("預算特徵檔缺少 OHLCV 欄位（需要 open/high/low/close/volume）以產生 time-driven 標籤")
        dfb = feat_df.loc[:, ohlcv_cols].copy()
        dfb = dfb.reindex(full_idx)
        feat_df = feat_df.reindex(full_idx)

        # 篩選被 enable 的特徵（以 plan + OHLCV + 1-min）
        feat_cols = select_plan_columns(feat_df, cfg)
        drop_feat = [c for c in feat_df.columns if c not in set(feat_cols)]
        if drop_feat:
            print(f"[INFO] Dropping {len(drop_feat)} precomputed cols not enabled: {drop_feat[:10]}{' ...' if len(drop_feat)>10 else ''}")
        if not feat_cols:
            raise ValueError("計畫啟用的特徵在預算檔中皆不存在（feat_cols 為空），請檢查 plan 與預算欄位。")
        feat_df = feat_df.loc[:, feat_cols].astype(np.float32)

        is_reg = (task_type == "regression")
        y_series = create_label(dfb, cfg, return_what=("reg" if is_reg else "cls"))

        # 清理與對齊
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
        valid_now = feat_df.notna().all(axis=1)
        valid_lbl = y_series.notna()
        keep = valid_now & valid_lbl
        feat_df = feat_df.loc[keep]
        y_series = y_series.loc[keep]

        # 時間區間篩選
        cv_start = pd.Timestamp(cfg["cv"]["start_date"]).tz_localize("UTC")
        cv_end   = pd.Timestamp(cfg["cv"]["end_date"]).tz_localize("UTC")
        cv_end   = cv_end + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        mask_range = (feat_df.index >= cv_start) & (feat_df.index <= cv_end)
        feat_df = feat_df.loc[mask_range]
        y_series = y_series.loc[mask_range]

        # 構造 df 與 meta
        df = pd.concat([feat_df, y_series], axis=1)
        feat_cols = list(feat_df.columns)
        target_col = "y_reg" if is_reg else "y_cls"
    is_reg = (task_type == "regression")

    # Scaler（只 fit 在 train，否則會洩漏）
    scaler_kind = cfg["sequence"]["scaler"]
    scaler_window = cfg["sequence"]["seq_len"]
    min_frac = cfg["sequence"]["min_frac"]

    # 1) 依 fold（基於 ref_index 的布林遮罩）映射到目前 df.index
    #    先將布林遮罩轉成時間集合，再以 isin 到當前 df.index 取得對應位置
    local_index = pd.DatetimeIndex(df.index)
    tv_times = ref_index[np.asarray(fold["train_val_mask"]).astype(bool)]
    te_times = ref_index[np.asarray(fold["test_mask"]).astype(bool)]
    tv_mask_local = local_index.isin(tv_times)
    te_mask_local = local_index.isin(te_times)
    df_tv_index = local_index[tv_mask_local]
    df_te_index = local_index[te_mask_local]

    split_ratio = cfg["cv"]["train_val_split"]
    split_pos = int(len(df_tv_index) * split_ratio)
    tr_idx = df_tv_index[:split_pos]
    va_idx = df_tv_index[split_pos:]
    te_idx = df_te_index

    # 1b) 決定要縮放的欄位（自動跳過 sign-like / 命名 pattern）
    cols_to_scale = pick_cols_to_scale(df.loc[tr_idx, feat_cols], feat_cols)

    # 2) 建立縮放器
    scaler = _get_scaler(scaler_kind, window=scaler_window, min_frac=min_frac)

    # 3) 時間安全縮放：先對整段 df 做 transform_full（只動 cols_to_scale）
    if hasattr(scaler, "is_timesafe") and scaler.is_timesafe:
        df_scaled = scaler.transform_full(df, cols_to_scale=cols_to_scale)
        work_df = df_scaled
        sklearn_scaler = None
    else:
        work_df = df
        sklearn_scaler = None if scaler is None else ColumnSubsetScaler(
            scaler, all_cols=feat_cols, cols_to_scale=cols_to_scale
        )


    # 4) 切 train/val/test（在時間安全模式已經先轉好；sklearn 模式稍後 fit+transform）
    
    X_tr, y_tr = work_df.loc[tr_idx, feat_cols], work_df.loc[tr_idx, target_col]
    X_va, y_va = work_df.loc[va_idx, feat_cols], work_df.loc[va_idx, target_col]
    X_te, y_te = work_df.loc[te_idx, feat_cols], work_df.loc[te_idx, target_col]

    def _clean_split(X_df, y_s):
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        valid = X_df.notna().all(axis=1) & y_s.notna()
        return X_df.loc[valid], y_s.loc[valid]

    # 清理nan
    X_tr, y_tr = _clean_split(X_tr, y_tr)
    X_va, y_va = _clean_split(X_va, y_va)
    X_te, y_te = _clean_split(X_te, y_te)

    # 5) sklearn 縮放：只用 train 擬合，SeqDataset 會在 GPU 前再 transform（不洩漏、且只動 cols_to_scale）
    if sklearn_scaler is not None:
        sklearn_scaler.fit_df(X_tr)

     # 6) 建 Dataset / Loader（和你原本一致）
    L = int(cfg["sequence"]["seq_len"])
    label_dtype = "float" if is_reg else "long"
    runtime_device = cfg["device"]
    bs = int(cfg["train"]["batch_size"])

    # 是否預先把整個 Dataset 放上 GPU（容易 OOM）；預設 False → 留在 CPU，再在 trainer 逐 batch 搬到 GPU
    preload_to_gpu = bool(cfg.get("sequence", {}).get("preload_to_gpu", False))
    ds_device = runtime_device if (preload_to_gpu and runtime_device == "cuda") else "cpu"

    stride = cfg["sequence"]["stride"]
    anchor = int(cfg["sequence"]["stride_anchor"]) % stride

    ds_tr = SeqDataset(X_tr, y_tr, L, scaler=sklearn_scaler, device=ds_device, label_dtype=label_dtype, stride=stride, anchor=anchor)
    ds_va = SeqDataset(X_va, y_va, L, scaler=sklearn_scaler, device=ds_device, label_dtype=label_dtype, stride=stride, anchor=anchor)
    ds_te = SeqDataset(X_te, y_te, L, scaler=sklearn_scaler, device=ds_device, label_dtype=label_dtype, stride=stride, anchor=anchor)

    # DataLoader：若 Dataset 在 CPU 且 runtime 在 CUDA，開啟 pin_memory + num_workers 加速搬運
    pin = (ds_device == "cpu" and runtime_device == "cuda")
    num_workers = 10 if pin else 0
    train_loader = DataLoader(ds_tr, batch_size=bs, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(ds_va, batch_size=bs, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(ds_te, batch_size=bs, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin)

    info = {"feat_cols": feat_cols, "target_col": target_col}


    # 7) 也把縮放資訊給 XGB 分支
    if also_XGB:
        Xtr = X_tr.values.astype(np.float32, copy=False)
        Xva = X_va.values.astype(np.float32, copy=False)
        Xte = X_te.values.astype(np.float32, copy=False)
        if sklearn_scaler is not None:
            Xtr = sklearn_scaler.transform(Xtr)
            Xva = sklearn_scaler.transform(Xva)
            Xte = sklearn_scaler.transform(Xte)
        y_dtype = np.float32 if is_reg else np.int64
        info["XGB"] = {
            "X_tr": Xtr, "y_tr": y_tr.values.astype(y_dtype, copy=False),
            "X_va": Xva, "y_va": y_va.values.astype(y_dtype, copy=False),
            "X_te": Xte, "y_te": y_te.values.astype(y_dtype, copy=False),
            "scaler": sklearn_scaler,
            "cols_to_scale": cols_to_scale,
        }

    # Optional: print binary label distribution for standard classification
    if (not is_reg):
        try:
            def _vc(s):
                vc = s.value_counts().to_dict()
                vc.setdefault(0, 0); vc.setdefault(1, 0)
                return {int(k): int(v) for k, v in vc.items() if int(k) in (0,1)}
            lbl_tr = _vc(y_tr)
            lbl_va = _vc(y_va)
            lbl_te = _vc(y_te)
            tm = fold.get("test_month", str(""))
            print(f"[Fold] test_month={tm} | label_counts: TR={lbl_tr} VA={lbl_va} TE={lbl_te}")
            info["label_counts"] = {"train": lbl_tr, "val": lbl_va, "test": lbl_te}
        except Exception:
            pass

    return train_loader, val_loader, test_loader, info


def make_event_loaders_for_fold(df_events: pd.DataFrame,
                                feat_cols: List[str],
                                fold: Dict,
                                cfg: Dict,
                                also_XGB: bool = False,
                                pre_feat_df: pd.DataFrame | None = None):
    """
    Event-driven loader builder using EventDataset:
    - Loads precomputed features (15m grid) from cfg.features.precomputed.path.
    - Applies scaler (time-safe rolling/ewm, or sklearn fit on train windows only).
    - Splits events by t0 alignment time using the given fold masks on df_events.index.
    """

    # === Split event index (t0 times) into train/val/test ===
    tr_idx, va_idx, te_idx = split_fold_to_indices(df_events, fold, cfg)

    # === Load full-grid features from precomputed file only ===
    if pre_feat_df is not None:
        feat_df = pre_feat_df.copy()
    else:
        pre_path = cfg["data"]["path"]
        if not pre_path:
            raise ValueError("event 模式需要 config.features.precomputed.path 指定預算特徵檔")
        p = str(pre_path)
        if p.endswith(".csv"):
            feat_df = pd.read_csv(p)
        elif p.endswith(".parquet"):
            feat_df = pd.read_parquet(p)
        else:
            raise ValueError("features.precomputed.path 只支援 .csv 或 .parquet")
        if "datetime" in feat_df.columns:
            idx = pd.to_datetime(feat_df["datetime"], errors="coerce", utc=True)
            feat_df = feat_df.drop(columns=["datetime"]) 
            feat_df.index = idx
        elif "timestamp" in feat_df.columns:
            ts = pd.to_numeric(feat_df["timestamp"], errors="coerce").astype("Int64")
            unit = "ms" if (ts.dropna().iloc[0] if len(ts.dropna()) else 0) > 1_000_000_000_000 else "s"
            idx = pd.to_datetime(ts, unit=unit, utc=True)
            feat_df = feat_df.drop(columns=["timestamp"]) 
            feat_df.index = idx
        feat_df = feat_df.sort_index()
        feat_df = feat_df[~feat_df.index.duplicated(keep="last")]

    # Restrict to selected features only (ensure order)
    if not feat_cols:
        feat_cols = select_plan_columns(feat_df, cfg)
    feat_df = feat_df.loc[:, [c for c in feat_cols if c in feat_df.columns]].astype(np.float32)

    # === Build scaler ===
    scaler_kind = cfg["sequence"]["scaler"]
    scaler_window = int(cfg["sequence"]["seq_len"])  # reuse seq_len
    min_frac = float(cfg["sequence"]["min_frac"]) if "min_frac" in cfg["sequence"] else 0.2
    scaler = _get_scaler(scaler_kind, window=scaler_window, min_frac=min_frac)

    # Helper: compute align positions for a set of t0 times
    def align_times(t0_index: pd.DatetimeIndex, idx_all: pd.DatetimeIndex, method: str) -> pd.DatetimeIndex:
        method = str(method).lower()
        t0u = pd.DatetimeIndex(t0_index)
        if t0u.tz is None:
            t0u = t0u.tz_localize("UTC")
        else:
            t0u = t0u.tz_convert("UTC")
        if method == "exact":
            pos = idx_all.get_indexer(t0u)
            valid = pos >= 0
            pos = pos[valid]
            return idx_all[pos]
        elif method == "pad":
            pos = idx_all.searchsorted(t0u, side="right") - 1
            valid = pos >= 0
            pos = pos[valid]
            return idx_all[pos]
        else:
            raise ValueError("align_method must be 'exact' or 'pad'")

    align_method = str(cfg.get("label", {}).get("align_method", "pad")).lower()
    L = int(cfg["sequence"]["seq_len"])  # window length
    idx_all = pd.DatetimeIndex(feat_df.index)

    # === Choose columns to scale using training windows only ===
    # Build union of all bars used in train-event windows: [p-L, p)
    train_align = align_times(tr_idx, idx_all, align_method)
    fit_pos = []
    # map times to integer positions
    # pos_map = pd.Series(np.arange(len(idx_all)), index=idx_all)
    # for at in train_align:
    #     p = int(pos_map.get(at, -1))
    #     if p <= 0:
    #         continue
    #     start = max(0, p - L)
    #     fit_pos.extend(range(start, p))
    # fit_pos = np.unique(np.array(fit_pos, dtype=int))
    # fit_index = idx_all[fit_pos] if len(fit_pos) else train_align  # fallback: use align times

    # vectorized build of fit positions: union over [p-L, p)
    p_vec = idx_all.searchsorted(train_align, side="right") - 1  # [N_tr]
    p_vec = p_vec[p_vec >= 0]
    if len(p_vec):
        rng = np.arange(L, dtype=np.int32)
        fit_pos = (p_vec[:, None] - rng[None, :]).reshape(-1)
        fit_pos = fit_pos[(fit_pos >= 0) & (fit_pos < len(idx_all))]
        fit_pos = np.unique(fit_pos)
        fit_index = idx_all[fit_pos]
    else:
        fit_index = train_align

    cols_to_scale = pick_cols_to_scale(feat_df.loc[fit_index, feat_cols], feat_cols)

    # === Apply scaler ===
    if hasattr(scaler, "is_timesafe") and scaler.is_timesafe:
        feat_scaled = scaler.transform_full(feat_df, cols_to_scale=cols_to_scale)
    else:
        if scaler is None:
            feat_scaled = feat_df
            sklearn_scaler = None
        else:
            sklearn_scaler = ColumnSubsetScaler(scaler, all_cols=feat_cols, cols_to_scale=cols_to_scale)
            sklearn_scaler.fit_df(feat_df.loc[fit_index, feat_cols])
            arr = feat_df.loc[:, feat_cols].values.astype(np.float32, copy=False)
            arr = sklearn_scaler.transform(arr)
            feat_scaled = feat_df.copy()
            feat_scaled.loc[:, feat_cols] = arr

    # === Build three EventDataset partitions (preloaded on device) ===
    tbm_csv_path = cfg["label"]["tbm_csv_path"]
    keep_sides = str(cfg["label"].get("keep_sides", "both")).lower()
    runtime_device = cfg["device"]
    bs = int(cfg["train"]["batch_size"])

    tr_align = align_times(tr_idx, idx_all, align_method)
    va_align = align_times(va_idx, idx_all, align_method)
    te_align = align_times(te_idx, idx_all, align_method)

    ds_tr = EventDataset(feat_scaled, tbm_csv_path, seq_len=L,
                         feature_cols=feat_cols, keep_sides=keep_sides,
                         align_method=align_method, device=runtime_device,
                         allowed_align_index=tr_align)
    ds_va = EventDataset(feat_scaled, tbm_csv_path, seq_len=L,
                         feature_cols=feat_cols, keep_sides=keep_sides,
                         align_method=align_method, device=runtime_device,
                         allowed_align_index=va_align)
    ds_te = EventDataset(feat_scaled, tbm_csv_path, seq_len=L,
                         feature_cols=feat_cols, keep_sides=keep_sides,
                         align_method=align_method, device=runtime_device,
                         allowed_align_index=te_align)

    # DataLoader（資料已在目標裝置，故不需 pin_memory）
    train_loader = DataLoader(ds_tr, batch_size=bs, shuffle=False, drop_last=False, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(ds_va, batch_size=bs, shuffle=False, drop_last=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(ds_te, batch_size=bs, shuffle=False, drop_last=False, num_workers=0, pin_memory=False)

    # Label distribution per split (0/1 counts)
    def _counts(ds):
        import torch
        y = ds.y
        if isinstance(y, torch.Tensor):
            y = y.detach().to("cpu")
            uniq, cnt = torch.unique(y, return_counts=True)
            d = {int(u.item()): int(c.item()) for u, c in zip(uniq, cnt)}
        else:
            import numpy as np
            arr = np.asarray(y)
            u, c = np.unique(arr, return_counts=True)
            d = {int(uu): int(cc) for uu, cc in zip(u, c)}
        # Ensure keys 0/1 exist
        d.setdefault(0, 0); d.setdefault(1, 0)
        return d

    lbl_tr, lbl_va, lbl_te = _counts(ds_tr), _counts(ds_va), _counts(ds_te)
    tm = fold.get("test_month", str(""))
    print(f"[EventFold] test_month={tm} | label_counts: TR={lbl_tr} VA={lbl_va} TE={lbl_te}")

    info = {"feat_cols": feat_cols, "target_col": "label", "label_counts": {"train": lbl_tr, "val": lbl_va, "test": lbl_te}}

    # XGB pack（flatten sequences；僅為滿足流程，分類時通常不使用）
    if also_XGB:
        def as_np(x):
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().to("cpu").numpy()
            return np.asarray(x)
        Xtr = as_np(ds_tr.X).reshape(len(ds_tr), -1).astype(np.float32, copy=False)
        Xva = as_np(ds_va.X).reshape(len(ds_va), -1).astype(np.float32, copy=False)
        Xte = as_np(ds_te.X).reshape(len(ds_te), -1).astype(np.float32, copy=False)
        ytr = as_np(ds_tr.y).astype(np.int64, copy=False)
        yva = as_np(ds_va.y).astype(np.int64, copy=False)
        yte = as_np(ds_te.y).astype(np.int64, copy=False)
        info["XGB"] = {
            "X_tr": Xtr, "y_tr": ytr,
            "X_va": Xva, "y_va": yva,
            "X_te": Xte, "y_te": yte,
            "scaler": None,
            "cols_to_scale": [],
        }

    return train_loader, val_loader, test_loader, info

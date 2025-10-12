# train/data/folds.py
"""
Cross-validation（時間序列）折疊工具。

本模組提供：
1) FoldGenerator：根據時間索引（DatetimeIndex）產生 rolling 或 purged K-fold 的
   train/val/test 分割，並可配合 cv.start_date / cv.end_date 內縮可用資料區間。
2) split_fold_to_indices：將 FoldGenerator 產生的 fold（可為布林遮罩或時間集合）
   映射成實際的 train/val/test 三段 DatetimeIndex。

特色
----
- 一律在內部統一使用「UTC tz-aware」來避免時區混亂。
- 支援以月份（Period[M]）為單位的測試月份序列（M、Q、或 2M/3M 步長）。
- Purged K-fold 會在測試區間前/後施加 embargo（禁運期），避免洩漏。
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


__all__ = ["FoldGenerator", "split_fold_to_indices"]


class FoldGenerator:
    """
    以時間索引建立交叉驗證折疊的產生器。

    參數
    ----
    dt_index : pd.DatetimeIndex
        原始時間索引（可為 naive 或 tz-aware）。內部會統一轉為 UTC tz-aware。
    mode : str, 預設 "rolling"
        保留以維持既有介面；目前不影響行為。
    start_month : str | None
        限定可用資料的起始日期（含當日），例如 "2023-01-01"；None 則以資料最早時間。
    end_month : str | None
        限定可用資料的結束日期（含當日），例如 "2025-04-30"；None 則以資料最晚時間。
    **kwargs
        保留以維持相容性。

    屬性
    ----
    months : list[pd.Period]
        於起訖範圍內的月份序列（Period[M]，naive）。
    start_ts_aware / end_ts_aware : pd.Timestamp
        內縮後的 UTC aware 起訖邊界；end 設為當日 23:59:59.999999999（含訖）。
    """

    def __init__(
        self,
        dt_index: pd.DatetimeIndex,
        mode: str = "rolling",
        start_month: str | None = None,
        end_month: str | None = None,
        **kwargs,
    ):
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
        self.start_ts_naive = (
            pd.Timestamp(start_month)
            if start_month is not None
            else pd.Timestamp(dt_index.min().tz_convert("UTC").tz_localize(None))
        )
        self.end_ts_naive = (
            pd.Timestamp(end_month)
            if end_month is not None
            else pd.Timestamp(dt_index.max().tz_convert("UTC").tz_localize(None))
        )

        # 對齊到「日期存在」且 start<=end
        if self.end_ts_naive < self.start_ts_naive:
            raise ValueError(
                f"[FoldGenerator] end_month({self.end_ts_naive}) 早於 start_month({self.start_ts_naive})"
            )

        # aware（UTC）
        self.start_ts_aware = self.start_ts_naive.tz_localize("UTC")
        # end 設為當日 23:59:59.999999999（含訖），避免月底被排除
        self.end_ts_aware = (
            self.end_ts_naive + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        ).tz_localize("UTC")

        # 3) 先將 dt_index 限縮到 [start,end]（保證後續 months 與遮罩一致）
        in_range = (self.dt_index >= self.start_ts_aware) & (self.dt_index <= self.end_ts_aware)
        self.dt_index = self.dt_index[in_range]
        if len(self.dt_index) == 0:
            raise ValueError("[FoldGenerator] 篩選後的 dt_index 為空，請檢查 cv.start_date / cv.end_date")

        # 4) 建立月份清單（用 naive；PeriodIndex 皆為 naive）
        dt_index_naive = self.dt_index.tz_convert("UTC").tz_localize(None)
        months_all = pd.PeriodIndex(dt_index_naive.to_period("M")).unique().sort_values()
        m0 = self.start_ts_naive.to_period("M")
        m1 = self.end_ts_naive.to_period("M")
        # 用 Period 直接比較，確保包含 start 所在之月份
        self.months = [m for m in months_all if (m >= m0) and (m <= m1)]

    def _get_test_months(self, test_freq: str):
        """
        產生測試月份序列。

        參數
        ----
        test_freq : str
            測試頻率。可為 'M'、'Q'，或 '2M'/'3M' 等步長。

        回傳
        ----
        pd.PeriodIndex | list[pd.Period]
            測試月份序列（限制在內縮後的 [start, end] 範圍）。
        """
        if not self.months:
            return []
        if test_freq == "M":
            # 每月一筆，直接回 self.months 即可
            return list(self.months)
        elif test_freq == "Q":
            # 以 self.months[0] 對齊，每 3 個月取一次
            first = self.months[0].month
            return [m for m in self.months if ((m.month - first) % 3 == 0)]
        else:
            # "2M","3M"…等步長
            step = int(test_freq.replace("M",""))
            return self.months[::step]

    # =========================
    # Rolling folds（原行為維持；僅用內縮後的 months 與 aware 邊界）
    # =========================
    def make_rolling_folds(self, train_window: int, embargo_hours: int, test_freq: str = "M") -> List[Dict]:
        """
        依固定訓練窗大小與測試頻率產生 rolling folds。

        參數
        ----
        train_window : int
            訓練窗涵蓋的「月份數」。
        embargo_hours : int
            訓練與測試的時間間隔（小時）以避免洩漏。
        test_freq : str, 預設 'M'
            測試月份頻率（'M'、'Q'、或 '2M'/'3M' 等）。

        回傳
        ----
        list[dict]
            每折包含：
            - 'train_val_mask' / 'test_mask'：對 self.dt_index 的布林遮罩
            - 'test_month'：該折測試月份（字串）
            - 'train_val_times' / 'test_times'：對應的時間集合（DatetimeIndex）
        """
        test_months = self._get_test_months(test_freq)
        folds: List[Dict] = []
        month_periods = pd.PeriodIndex(self.dt_index.tz_convert("UTC").tz_localize(None).to_period("M"))

        for m in test_months:
            i = self.months.index(m)
            if i < train_window or i + 1 >= len(self.months):
                continue

            train_start = pd.Timestamp(self.months[i - train_window].start_time, tz="UTC")
            test_start = pd.Timestamp(m.start_time, tz="UTC")
            embargo_delta = pd.Timedelta(hours=embargo_hours)
            train_end = test_start - embargo_delta

            # 也限制在 end_ts_aware 以內
            train_mask = (self.dt_index >= max(train_start, self.start_ts_aware)) & (
                self.dt_index < min(train_end, self.end_ts_aware)
            )
            test_mask = (month_periods == m) & (self.dt_index <= self.end_ts_aware)

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            folds.append(
                {
                    "train_val_mask": train_mask,
                    "test_mask": test_mask,
                    "test_month": str(m),
                    # 新增：時間集合（對應 bar 級 self.dt_index）
                    "train_val_times": self.dt_index[train_mask],
                    "test_times": self.dt_index[test_mask],
                }
            )
        return folds

    # =========================
    # Purged K-fold（含 embargo；同時套用起訖）
    # =========================
    def make_purged_kfold(
        self,
        n_splits: int = 5,
        embargo_hours: int = 24,
        min_train_days: int = 30,
    ) -> List[Dict]:
        """
        以「時間」切分的 Purged K-Fold（含 embargo）。

        流程
        ----
        - months 先依 cv.start_date / cv.end_date 內縮；
        - 將 months 均分為 n_splits 區段，取每段最後一月為測試月（如 10 個月/5 折 → 2,4,6,8,10）。
        - 訓練集 = 全資料中，剔除 [test_start - embargo, test_end + embargo) 的樣本，
          並限制在 [start_date, end_date] 範圍內。

        參數
        ----
        n_splits : int, 預設 5
            折數（需 >=2）。
        embargo_hours : int, 預設 24
            禁運期（小時）。
        min_train_days : int, 預設 30
            訓練天數下限（不符合者會跳過該折）。

        回傳
        ----
        list[dict]
            每折包含 'train_val_mask'、'test_mask'、'test_month' 與時間集合。
        """
        assert n_splits >= 2, "[make_purged_kfold] n_splits 需 >= 2"

        dt_naive = self.dt_index.tz_convert("UTC").tz_localize(None)
        month_periods = pd.PeriodIndex(dt_naive.to_period("M"))
        embargo = pd.Timedelta(hours=int(embargo_hours))

        m_idx = np.arange(len(self.months))
        groups = np.array_split(m_idx, n_splits)

        folds: List[Dict] = []
        for g in groups:
            if len(g) == 0:
                continue
            test_m = self.months[g[-1]]
            test_mask = (month_periods == test_m) & (self.dt_index <= self.end_ts_aware)
            if test_mask.sum() == 0:
                continue

            test_start = pd.Timestamp(test_m.start_time, tz="UTC")
            test_end = pd.Timestamp((test_m + 1).start_time, tz="UTC")
            purge_start, purge_end = test_start - embargo, test_end + embargo

            # 限制在 [start_aware, end_aware] & Purge
            base = (self.dt_index >= self.start_ts_aware) & (self.dt_index <= self.end_ts_aware)
            no_overlap = (self.dt_index < purge_start) | (self.dt_index >= purge_end)
            train_mask = base & no_overlap & (~test_mask)

            if train_mask.sum() == 0:
                continue

            tr_days = (
                pd.DatetimeIndex(self.dt_index[train_mask]).date[-1]
                - pd.DatetimeIndex(self.dt_index[train_mask]).date[0]
            ).days + 1
            if tr_days < int(min_train_days):
                continue

            folds.append(
                {
                    "train_val_mask": train_mask,
                    "test_mask": test_mask,
                    "test_month": str(test_m),
                    # 新增：時間集合（對應 bar 級 self.dt_index）
                    "train_val_times": self.dt_index[train_mask],
                    "test_times": self.dt_index[test_mask],
                }
            )
        return folds

# -------------------------
def split_fold_to_indices(df: pd.DataFrame, fold: Dict, cfg: Dict):
    """
    依 fold 切出 train/val/test 的索引（DatetimeIndex）。

    行為
    ----
    - 若 fold 的布林遮罩長度與 df 相同，直接使用布林遮罩（time-driven 常見）。
    - 否則改用 fold['train_val_times'] / fold['test_times'] 做「時間對齊」
     （event-driven 推薦／亦可做為備援）。
    - 回傳三段 DatetimeIndex（已按時間排序），並依 cfg['cv']['train_val_split'] 切 train/val。

    參數
    ----
    df : pd.DataFrame
        目標資料（可能是 bar 級，或事件 t0 級）；僅使用其 index。
    fold : dict
        FoldGenerator 產生的折疊資訊。
    cfg : dict
        需包含 'cv.train_val_split'（0~1）。

    回傳
    ----
    (tr_idx, va_idx, te_idx) : Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]
        三段索引，分別對應 train / val / test。
    """
    train_val_mask = fold.get("train_val_mask")
    test_mask = fold.get("test_mask")

    # 標準化 df.index → UTC tz-aware
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")

    # 判斷是否可直接用布林遮罩
    use_mask_direct = (
        isinstance(train_val_mask, (np.ndarray, pd.Series))
        and isinstance(test_mask, (np.ndarray, pd.Series))
        and len(train_val_mask) == len(df)
        and len(test_mask) == len(df)
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

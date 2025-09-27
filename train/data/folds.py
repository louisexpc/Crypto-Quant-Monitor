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

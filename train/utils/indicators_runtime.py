# === indicators.py ===
from __future__ import annotations
import json, hashlib
from pathlib import Path
from typing import Dict, Any, Union

import numpy as np
import pandas as pd
import pandas_ta as ta


# 對齊論文 Top-8 的預設方案
PAPER_TOP8_PLAN: Dict[str, Any] = {
    "features": [
        {"name": "RSI",   "kwargs": {"length": 30},  "enabled": True},   # RSI30
        {"name": "MACD",  "kwargs": {"fast": 12, "slow": 26, "signal": 9}, "enabled": True},
        {"name": "MOM",   "kwargs": {"length": 30},  "enabled": True},   # MOM30
        {"name": "STOCH", "kwargs": {"k": 30,  "d": 3, "smooth_k": 3}, "enabled": True},   # %K30,%D30
        {"name": "STOCH", "kwargs": {"k": 200, "d": 3, "smooth_k": 3}, "enabled": True},  # %K200,%D200
        {"name": "RSI",   "kwargs": {"length": 14},  "enabled": True},   # RSI14
    ]
}

# 覆蓋重複、涵蓋趨勢/動能/震盪/波動/量價的最佳20指標
BEST20_PLAN: Dict[str, Any] = {
    "features": [
        {"name": "DPO",       "kwargs": {"length": 20},                "enabled": True},
        {"name": "PVO",       "kwargs": {"fast": 12, "slow": 26, "signal": 9}, "enabled": True},
        {"name": "VOLUME",    "kwargs": {},                            "enabled": True},
        {"name": "BOP",       "kwargs": {},                            "enabled": True},
        {"name": "BBP",       "kwargs": {"length": 5, "std": 2.0},     "enabled": True},
        {"name": "MASSI",     "kwargs": {"fast": 9, "slow": 25},       "enabled": True},
        {"name": "KDJ",       "kwargs": {"k": 9, "d": 3, "smooth_k": 3}, "enabled": True},  # 取 J 線
        {"name": "TTM_TRND",  "kwargs": {"length": 6},                 "enabled": True},
        {"name": "WILLR",     "kwargs": {"length": 14},                "enabled": True},
        {"name": "LOGRET",    "kwargs": {"length": 1},                 "enabled": True},
        {"name": "UO",        "kwargs": {"fast": 7, "medium": 14, "slow": 28}, "enabled": True},
        {"name": "RVI",       "kwargs": {"length": 14},                "enabled": True},
        {"name": "TRUERANGE", "kwargs": {},                            "enabled": True},    # TRUERANGE_1
        {"name": "AMATE_LR",  "kwargs": {"fast": 8, "slow": 21, "mamode": 2}, "enabled": True},
        {"name": "SLOPE",     "kwargs": {"length": 1},                 "enabled": True},
        {"name": "EBSW",      "kwargs": {"length": 40, "mamode": 10},  "enabled": True},
        {"name": "CCI",       "kwargs": {"length": 14, "c": 0.015},    "enabled": True},
        {"name": "ZS",        "kwargs": {"length": 30},                "enabled": True},
        {"name": "RSI",       "kwargs": {"length": 14},                "enabled": True},
        {"name": "PVR",       "kwargs": {},                            "enabled": True},
    ]
}


class Indicators:
    """
    用法：
      df_raw = pd.read_csv("btc_15m.csv")          # <- 外面自己讀
      ind = Indicators(df_raw, freq_check="15T")   # <- 交給 class，內部會規格化 self.df
      X   = ind.compute(Indicators.PAPER_TOP8_PLAN)

    特點：
      - 內部保存 self.df（規格化後的 OHLCV），所有指標計算用 self.df
      - 規格化包含：建 UTC DatetimeIndex、排序、去重、只留 open/high/low/close/volume、轉 float32
      - 指紋快取：以 (資料指紋 + 計畫指紋) 做 parquet 快取
      - 防洩漏：所有特徵統一 shift(1)
    """

    def __init__(self,
                 df_raw: pd.DataFrame,
                 cache_dir: Union[str, Path] = "cache_features",
                 freq_check: str | None = None,
                 prefer_time_col: str = "timestamp"):
        """
        df_raw: 外部讀進來的 DataFrame（可含 datetime/timestamp 欄位或已是 DatetimeIndex）
        cache_dir: 指標快取輸出目錄（parquet）
        freq_check: 例如 "15T"；若提供則只做缺口提醒，不自動補齊
        prefer_time_col: 當 df_raw 不是 DatetimeIndex 時，優先用哪個時間欄位建索引
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.freq_check = freq_check
        self.prefer_time_col = prefer_time_col

        # 規格化並存成 self.df（後續所有計算都用它）
        self.df = self._normalize_ohlcv(df_raw)

    # ---------- 規格化：建 UTC DatetimeIndex + 只留 OHLCV ----------
    def _normalize_ohlcv(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        df.columns = [c.lower() for c in df.columns]

        # 0) 若已是 DatetimeIndex + 有時區，就直接用；否則嘗試從欄位建 index
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
            idx = None
            # 先試用 prefer_time_col
            if self.prefer_time_col in df.columns:
                idx = self._make_utc_index_from_col(df[self.prefer_time_col], self.prefer_time_col)
            else:
                # 退而求其次：timestamp/datetime
                if "timestamp" in df.columns:
                    idx = self._make_utc_index_from_col(df["timestamp"], "timestamp")
                elif "datetime" in df.columns:
                    idx = self._make_utc_index_from_col(df["datetime"], "datetime")
                else:
                    raise ValueError("需要 DatetimeIndex 或 'timestamp'/'datetime' 欄位來建立索引")

            df.index = idx

        # 1) 排序、去重
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # 2) 只留 OHLCV 並轉 dtype
        cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"缺少欄位: {missing}")
        df = df[cols].astype("float32")

        # 3) 可選：檢查頻率缺口
        if self.freq_check and len(df) > 1:
            expected = pd.date_range(df.index[0], df.index[-1], freq=self.freq_check, tz="UTC")
            miss = expected.difference(df.index)
            if len(miss) > 0:
                print(f"[WARN] 缺少 {len(miss)} 根 {self.freq_check} K；預設不補齊。")

        return df
    
    @staticmethod
    def _make_utc_index_from_col(col: pd.Series, kind: str) -> pd.DatetimeIndex:
        if kind == "timestamp":
            ts = col.astype("int64")
            unit = "ms" if ts.iloc[0] > 1_000_000_000_000 else "s"
            return pd.to_datetime(ts, unit=unit, utc=True)
        elif kind == "datetime":
            return pd.to_datetime(col, utc=True)
        else:
            raise ValueError(f"未知時間欄位型態：{kind}")
        
    # === 放進 Indicators 類別裡 ===
    def _leak_check_single(self, builder_fn, name: str, kwargs: dict, n_cuts: int = 4, seed: int = 42):
        """
        檢查單一指標是否偷看未來（prefix invariance test）
        步驟：對若干個隨機 cut t，分別用 df[:t] 與 df[:N] 計算，比較在 (t-1) 的值是否一致（考慮 shift(1)）
        回傳 (ok: bool, msg: str|None)
        """
        rng = np.random.default_rng(seed)
        N = len(self.df)
        if N < 200:
            return True, None  # 太短不測

        # 隨機選擇幾個 cut（避開太前/太後，保留窗口）
        cuts = rng.integers(low=int(N*0.3), high=int(N*0.9), size=n_cuts)

        # 全長版本（作為參考）
        full = builder_fn(kwargs)
        full = full.shift(1)  # 與 compute() 保持一致
        full = full.replace([np.inf, -np.inf], np.nan)

        if not isinstance(full, pd.DataFrame):
            full = pd.DataFrame(full)

        colnames = list(full.columns)

        for t in cuts:
            # 截斷到 t 的版本
            df_trunc = self.df.iloc[:t].copy()
            # 用相同 builder，但要能在截斷 df 上跑；因此 builder_fn 需關聯 self.df
            # 我們暫時把 self.df 換掉，跑完再換回來
            df_backup = self.df
            try:
                self.df = df_trunc
                trunc = builder_fn(kwargs).shift(1).replace([np.inf, -np.inf], np.nan)
                if not isinstance(trunc, pd.DataFrame):
                    trunc = pd.DataFrame(trunc)
            finally:
                self.df = df_backup

            # 比較第 (t-1) 的值（該點若需要未來才有值，兩者會不同）
            idx = full.index[min(t-1, len(full)-1)]
            for c in colnames:
                v_full  = full.loc[idx, c] if c in full.columns else np.nan
                v_trunc = trunc.loc[idx, c] if (c in trunc.columns and idx in trunc.index) else np.nan
                # 允許 NaN 相等
                if pd.isna(v_full) and pd.isna(v_trunc):
                    continue
                # 允許極小數值誤差
                if not (pd.notna(v_full) and pd.notna(v_trunc) and np.allclose(v_full, v_trunc, atol=1e-8, rtol=1e-5)):
                    return False, f"[LEAK?] {name} at {idx} col={c}: full={v_full} vs trunc={v_trunc}"

        return True, None

    # ---------- 指紋（資料 + 計畫） ----------
    @staticmethod
    def _fingerprint_df(df_ohlcv: pd.DataFrame) -> str:
        sig = pd.Series({
            "rows": int(len(df_ohlcv)),
            "idx0": str(df_ohlcv.index[0]) if len(df_ohlcv) else "NA",
            "idxN": str(df_ohlcv.index[-1]) if len(df_ohlcv) else "NA",
            "sum_close": float(np.nan_to_num(df_ohlcv["close"].sum())),
            "sum_vol": float(np.nan_to_num(df_ohlcv["volume"].sum())),
        })
        return hashlib.md5(sig.to_json().encode()).hexdigest()[:10]

    @staticmethod
    def _fingerprint_plan(plan: Dict[str, Any]) -> str:
        payload = json.dumps(plan, sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()[:10]

    def _cache_path(self, df_fp: str, plan_fp: str) -> Path:
        return self.cache_dir / f"feat_{df_fp}_{plan_fp}.parquet"
    
    # ---------- 指標 builders（最小集合：RSI, MACD, MOM, STOCH） ----------
    @staticmethod
    def _build_RSI(df: pd.DataFrame, length: int)-> pd.DataFrame:
        out = ta.rsi(df["close"], length=length)
        return pd.DataFrame({f"RSI_{length}": out})
    
    @staticmethod
    def _build_MACD(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        macd = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
        col_main = macd.columns[0]  # 只取主 MACD
        return pd.DataFrame({f"MACD_{fast}_{slow}_{signal}": macd[col_main]})

    @staticmethod
    def _build_MOM(df: pd.DataFrame, length: int) -> pd.DataFrame:
        out = ta.mom(df["close"], length=length)
        return pd.DataFrame({f"MOM_{length}": out})
    
    @staticmethod
    def _build_STOCH(df: pd.DataFrame, k: int, d: int = 3, smooth_k: int = 3) -> pd.DataFrame:
        st = ta.stoch(df["high"], df["low"], df["close"], k=k, d=d, smooth_k=smooth_k)
        return st.rename(columns={
            st.columns[0]: f"STOCHk_{k}",
            st.columns[1]: f"STOCHd_{k}",
        })[[f"STOCHk_{k}", f"STOCHd_{k}"]]
    

    # === 20 指標對應的 builders ===
    @staticmethod
    def _build_DPO(df: pd.DataFrame, length: int) -> pd.DataFrame:
        s = ta.dpo(df["close"], length=length,centered=False)
        return pd.DataFrame({f"DPO_{length}": s})

    @staticmethod
    def _build_PVO(df: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
        pvo = ta.pvo(df["volume"], fast=fast, slow=slow, signal=signal)
        # 主線欄名通常形如 'PVO_12_26_9'
        col = [c for c in pvo.columns if c.startswith("PVO_") and c.count("_") >= 2]
        main = col[0] if col else pvo.columns[0]
        return pd.DataFrame({main: pvo[main]})

    @staticmethod
    def _build_BOP(df: pd.DataFrame) -> pd.DataFrame:
        s = ta.bop(df["open"], df["high"], df["low"], df["close"])
        return pd.DataFrame({"BOP": s})

    @staticmethod
    def _build_BBP(df: pd.DataFrame, length: int, std: float) -> pd.DataFrame:
        bb = ta.bbands(df["close"], length=length, std=std)
        # 百分位位置 'BBP'
        col = [c for c in bb.columns if c.startswith("BBP_")]
        tgt = col[0] if col else bb.columns[-1]
        # 統一欄名
        return pd.DataFrame({f"BBP_{length}_{std}": bb[tgt]})

    @staticmethod
    def _build_MASSI(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
        s = ta.massi(df["high"], df["low"], fast=fast, slow=slow)
        return pd.DataFrame({f"MASSI_{fast}_{slow}": s})

    @staticmethod
    def _build_KDJ(df: pd.DataFrame, k: int, d: int, smooth_k: int) -> pd.DataFrame:
        # 用 STOCH 近似 KDJ：J = 3*K - 2*D
        st = ta.stoch(df["high"], df["low"], df["close"], k=k, d=d, smooth_k=smooth_k)
        kcol, dcol = st.columns[0], st.columns[1]
        j = 3 * st[kcol] - 2 * st[dcol]
        return pd.DataFrame({f"J_{k}_{d}": j})

    @staticmethod
    def _build_TTM_TRND(df: pd.DataFrame, length: int = 6):
        tt = ta.ttm_trend(df["high"], df["low"], df["close"], length=length)
        return pd.DataFrame({f"TTM_TRND_{length}": tt})  # ← 保證是單欄 DataFrame

    @staticmethod
    def _build_WILLR(df: pd.DataFrame, length: int) -> pd.DataFrame:
        s = ta.willr(df["high"], df["low"], df["close"], length=length)
        return pd.DataFrame({f"WILLR_{length}": s})

    @staticmethod
    def _build_LOGRET(df: pd.DataFrame, length: int) -> pd.DataFrame:
        s = np.log(df["close"]).diff(length)
        return pd.DataFrame({f"LOGRET_{length}": s})

    @staticmethod
    def _build_UO(df: pd.DataFrame, fast: int, medium: int, slow: int) -> pd.DataFrame:
        s = ta.uo(df["high"], df["low"], df["close"], fast=fast, medium=medium, slow=slow)
        return pd.DataFrame({f"UO_{fast}_{medium}_{slow}": s})

    @staticmethod
    def _build_RVI(df: pd.DataFrame, length: int) -> pd.DataFrame:
        rvi = ta.rvi(df["close"], length=length)  # 不同版本欄名略有差，做保護
        if isinstance(rvi, pd.DataFrame):
            col = [c for c in rvi.columns if c.startswith("RVI")]
            tgt = col[0] if col else rvi.columns[0]
            return pd.DataFrame({f"RVI_{length}": rvi[tgt]})
        else:
            return pd.DataFrame({f"RVI_{length}": rvi})

    @staticmethod
    def _build_TRUERANGE(df: pd.DataFrame) -> pd.DataFrame:
        # True Range (單期)：max(high-low, |high-close_prev|, |low-close_prev|)
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            (df["high"] - df["low"]),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs()
        ], axis=1).max(axis=1)
        return pd.DataFrame({"TRUERANGE_1": tr})

    @staticmethod
    def _build_AMATE_LR(df: pd.DataFrame, fast: int, slow: int, mamode: int = 2) -> pd.DataFrame:
        # pandas_ta 的 AMAT 可能回傳 LR/SR 兩欄
        amat = ta.amat(df["close"], fast=fast, slow=slow, mamode=mamode)
        # 盡量抓到 LR 欄
        col = [c for c in amat.columns if "AMATe_LR" in c or "LR_" in c]
        tgt = col[0] if col else amat.columns[0]
        return pd.DataFrame({f"AMATe_LR_{fast}_{slow}_{mamode}": amat[tgt]})

    @staticmethod
    def _build_SLOPE(df: pd.DataFrame, length: int) -> pd.DataFrame:
        s = ta.slope(df["close"], length=length)
        return pd.DataFrame({f"SLOPE_{length}": s})

    @staticmethod
    def _build_EBSW(df: pd.DataFrame, length: int, mamode: int) -> pd.DataFrame:
        # EBSW 在 pandas_ta 中常見為 ebsw(close, length, mamode)
        e = ta.ebsw(df["close"], length=length, mamode=mamode)
        if isinstance(e, pd.DataFrame):
            col = [c for c in e.columns if c.startswith("EBSW_")]
            tgt = col[0] if col else e.columns[0]
            return pd.DataFrame({f"EBSW_{length}_{mamode}": e[tgt]})
        else:
            return pd.DataFrame({f"EBSW_{length}_{mamode}": e})

    @staticmethod
    def _build_CCI(df: pd.DataFrame, length: int, c: float = 0.015) -> pd.DataFrame:
        s = ta.cci(df["high"], df["low"], df["close"], length=length, c=c)
        return pd.DataFrame({f"CCI_{length}_{c}": s})

    @staticmethod
    def _build_ZS(df: pd.DataFrame, length: int) -> pd.DataFrame:
        s = ta.zscore(df["close"], length=length)
        return pd.DataFrame({f"ZS_{length}": s})

    @staticmethod
    def _build_PVR(df: pd.DataFrame) -> pd.DataFrame:
        # pandas_ta.pvr 以 close 與 volume 為基礎
        s = ta.pvr(df["close"], df["volume"])
        name = "PVR" if isinstance(s, pd.Series) else (s.columns[0] if len(s.columns) else "PVR")
        return pd.DataFrame({name: s if isinstance(s, pd.Series) else s.iloc[:, 0]})



    


    # ---------- 選擇要使用的指標 ----------
    def _prune_plan(self, plan: dict) -> dict:
        """
        回傳只包含 enabled=True 的有效計畫（保持原順序）
        - features 項若未提供 enabled，視為 True
        """
        feats = []
        for it in plan.get("features", []):
            enabled = it.get("enabled", True)
            if enabled:
                feats.append({"name": it["name"], "kwargs": it.get("kwargs", {})})
        return {"features": feats}
    
    @property
    def _builders(self):
        # 乾淨的 name → builder 對應
        return {
            "RSI":   lambda kw: self._build_RSI(self.df, **kw),
            "MACD":  lambda kw: self._build_MACD(self.df, **kw),
            "MOM":   lambda kw: self._build_MOM(self.df, **kw),
            "STOCH": lambda kw: self._build_STOCH(self.df, **kw),

            # 新增的指標
            "DPO":         lambda kw: self._build_DPO(self.df, **kw),
            "PVO":         lambda kw: self._build_PVO(self.df, **kw),
            "BOP":         lambda kw: self._build_BOP(self.df),
            "BBP":         lambda kw: self._build_BBP(self.df, **kw),
            "MASSI":       lambda kw: self._build_MASSI(self.df, **kw),
            "KDJ":         lambda kw: self._build_KDJ(self.df, **kw),          # J 線
            "TTM_TRND":    lambda kw: self._build_TTM_TRND(self.df, **kw),
            "WILLR":       lambda kw: self._build_WILLR(self.df, **kw),
            "LOGRET":      lambda kw: self._build_LOGRET(self.df, **kw),
            "UO":          lambda kw: self._build_UO(self.df, **kw),
            "RVI":         lambda kw: self._build_RVI(self.df, **kw),
            "TRUERANGE":   lambda kw: self._build_TRUERANGE(self.df),
            "AMATE_LR":    lambda kw: self._build_AMATE_LR(self.df, **kw),
            "SLOPE":       lambda kw: self._build_SLOPE(self.df, **kw),
            "EBSW":        lambda kw: self._build_EBSW(self.df, **kw),
            "CCI":         lambda kw: self._build_CCI(self.df, **kw),
            "ZS":          lambda kw: self._build_ZS(self.df, **kw),
            "PVR":         lambda kw: self._build_PVR(self.df),
        }

    # ---------- 主入口：計畫制計算 + 快取（用 self.df） ----------
    def compute(self, plan: Dict[str, Any]) -> pd.DataFrame:
        """
        依 plan 計算指標並快取；支援 features[*].enabled=True/False
        - 只會對 enabled=True 的特徵計算與快取
        """
        # 1) 過濾出有效（enabled）特徵
        eff_plan = self._prune_plan(plan)
        if len(eff_plan.get("features", [])) == 0:
            raise ValueError("計畫中沒有任何 enabled=True 的特徵，請確認 plan 或 Optuna 開關。")
        
        # 2) 以『資料指紋 + 有效計畫指紋』組合快取鍵
        df_fp = self._fingerprint_df(self.df)
        pl_fp = self._fingerprint_plan(plan)
        cpath = self._cache_path(df_fp, pl_fp)

        if cpath.exists():
            return pd.read_parquet(cpath)        
        
        # # 3) 動態計算
        # parts = []
        # for item in eff_plan["features"]:
        #     name   = item["name"].upper()
        #     kwargs = item.get("kwargs", {})
        #     if name not in self._builders:
        #         raise ValueError(f"Unknown indicator: {name}")
        #     f = self._builders[name](kwargs)

        #     # 防洩漏：統一 shift(1)
        #     parts.append(f.shift(1))


        parts = []
        for item in eff_plan["features"]:
            name   = item["name"].upper()
            kwargs = item.get("kwargs", {})
            if name not in self._builders:
                raise ValueError(f"Unknown indicator: {name}")

            builder = self._builders[name]  # 這是 lambda kw: self._build_XXX(self.df, **kw)

            # （可選）先跑防漏檢查：prefix invariance
            
            ok, msg = self._leak_check_single(builder, name, kwargs, n_cuts=4, seed=42)
            if not ok:
                raise RuntimeError(msg)

            f = builder(kwargs)

            # 一律防洩漏
            if isinstance(f, pd.Series):
                f = f.to_frame()
            # 確保只有單欄（保守）
            if f.shape[1] > 1:
                f = f.iloc[:, [0]]

            parts.append(f.shift(1))

        # 4) 合併/清理/快取
        feat = pd.concat(parts, axis=1)
        feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
        feat.to_parquet(cpath, index=True)
        return feat
    
    # 查詢快取路徑
    def cache_path_for(self, plan: dict) -> Path:
        eff = self._prune_plan(plan)
        df_fp = self._fingerprint_df(self.df)
        pl_fp = self._fingerprint_plan(eff)
        return self.cache_dir / f"feat_{df_fp}_{pl_fp}.parquet"
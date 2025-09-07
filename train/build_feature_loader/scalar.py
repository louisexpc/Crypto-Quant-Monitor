# scalar.py

"""
提供特徵縮放工具：
- ColumnSubsetScaler：只縮放指定欄位（sklearn 類型）

- TimeSafeScaler：時序安全（不看未來）的標準化器（rolling / ewm / robust_rolling）
  - 支援 min_frac 控制 rolling/robust 的 warm-up 需求
  - 提供 warmup_len() 讓上游可預估前段 NaN 長度

- _get_scaler：統一入口；回傳 sklearn 縮放器或 TimeSafeScaler
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

def _is_sign_like_col(s: pd.Series) -> bool:
    """
    判斷欄位是否屬於 sign-like（值幾乎只在 {-1,0,1}）— 若是則不縮放。
    僅用數值本身判斷：非 NaN 唯一值 ⊆ {-1, 0, 1} → 視為 sign-like。
    """
    if s is None:
        return False
    vals = pd.Series(s.dropna().unique())
    if vals.empty:
        return False
    try:
        ss = set(vals.astype(float).tolist())
    except Exception:
        return False
    return ss.issubset({-1.0, 0.0, 1.0})

def pick_cols_to_scale(df: pd.DataFrame, feat_cols: list[str]) -> list[str]:
    """自動過濾：只回傳需要縮放的欄位（sign-like 欄自動跳過）"""
    return [c for c in feat_cols if not _is_sign_like_col(df[c])]

class ColumnSubsetScaler:
    """
    將 sklearn 縮放器限制在部分欄位：
      - fit_df(df): 只用子欄位擬合
      - transform(X_np): 僅轉換子欄位，其他欄位原樣保留
    """
    def __init__(self, base_scaler, all_cols: list[str], cols_to_scale: list[str]):
        self.base = base_scaler
        self.all_cols = list(all_cols)
        self.cols_to_scale = list(cols_to_scale)
        self.idxs = [self.all_cols.index(c) for c in self.cols_to_scale]

    def fit_df(self, df: pd.DataFrame):
        if not self.cols_to_scale:
            return
        Xsub = df.loc[:, self.cols_to_scale].values.astype(np.float32, copy=False)
        self.base.fit(Xsub)

    def transform(self, X_np: np.ndarray) -> np.ndarray:
        if not self.cols_to_scale:
            return X_np.astype(np.float32, copy=False)
        X_np = X_np.astype(np.float32, copy=False)
        X_np[:, self.idxs] = self.base.transform(X_np[:, self.idxs]).astype(np.float32, copy=False)
        return X_np

class TimeSafeScaler:
    """
    時序安全縮放器（不看未來）：
    - rolling：滾動均值/標準差 z-score
    - ewm：指數加權均值/方差 z-score（幾乎沒有固定 warm-up）
    - robust_rolling：中位數/MAD 的 robust z-score

    特點
    ----
    1) 一律對「過去」統計量做 shift(1)，確保 t 只用 t-1 以前資訊。
    2) 透過 min_frac 控制 rolling/robust 的 min_periods → 決定 warm-up 長度。
    3) transform_full(...) 會回傳含有前段 NaN 的 DataFrame（**請在 split 後各自丟掉 NaN**）。

    參數
    ----
    mode : {"rolling","ewm","robust_rolling"}
    window : int，視窗長度（rolling/robust 用）
    min_frac : float，min_periods = round(window * min_frac)，控制 warm-up
    """
    def __init__(self, mode: str = "rolling", window: int = 96,
                 min_frac: float = 0.2):  # ★ 新增：決定 rolling/robust 的 min_periods 占比
        mode = str(mode).lower()
        assert mode in {"rolling", "ewm", "robust_rolling"}
        self.mode = mode
        self.window = int(window)
        self.min_frac = float(min_frac)
        self.is_timesafe = True


    def warmup_len(self) -> int:
        """
        回傳預期的 warm-up 長度（以列數計），供上游在 train split 開頭裁掉。
        - ewm：幾乎只因 shift(1) 造成 1 列 NaN → 回 1
        - rolling/robust_rolling：min_periods + 1（+1 來自 shift(1)）
        """
        if self.mode == "ewm":
            return 1  # ewm 本質上沒有固定窗口，shift(1) 只讓第一筆 NaN
        # rolling / robust_rolling
        minp = max(2, int(round(self.window * self.min_frac)))
        return minp + 1  # +1 因為我們都有 shift(1)

    def _rolling_z(self, X: pd.DataFrame) -> pd.DataFrame:
        w = self.window
        minp = max(2, int(round(w * self.min_frac)))  # ★ 用 min_frac 控制 warm-up
        mu = X.rolling(w, min_periods=minp).mean().shift(1)
        sd = X.rolling(w, min_periods=minp).std(ddof=0).shift(1)
        return (X - mu) / (sd.replace(0.0, np.nan) + 1e-8)

    def _ewm_z(self, X: pd.DataFrame) -> pd.DataFrame:
        span = self.window
        mu  = X.ewm(span=span, adjust=False).mean().shift(1)
        var = X.ewm(span=span, adjust=False).var(bias=False).shift(1)
        sd  = (var.clip(lower=0.0))**0.5
        return (X - mu) / (sd + 1e-8)

    def _robust_rolling(self, X: pd.DataFrame) -> pd.DataFrame:
        w = self.window
        minp = max(2, int(round(w * self.min_frac)))  # ★ 同 rolling
        med = X.rolling(w, min_periods=minp).median().shift(1)
        mad = (X - med).abs().rolling(w, min_periods=minp).median().shift(1)
        return (X - med) / (1.4826 * mad + 1e-8)

    def transform_full(self, df: pd.DataFrame, cols_to_scale: list[str]) -> pd.DataFrame:
        out = df.copy()
        if cols_to_scale:
            X = df.loc[:, cols_to_scale].astype(np.float32)
            if self.mode == "rolling":
                Z = self._rolling_z(X)
            elif self.mode == "ewm":
                Z = self._ewm_z(X)
            else:
                Z = self._robust_rolling(X)
            out.loc[:, cols_to_scale] = Z.astype(np.float32)
        return out

# --- 統一入口 ---
def _get_scaler(scaler_kind: str, *, window: int = 96, min_frac: float = 0.2):
    kind = str(scaler_kind).lower()
    if kind == "standard":
        return StandardScaler()
    elif kind == "robust":
        return RobustScaler(with_centering=True, with_scaling=True, quantile_range=(10.0, 90.0))
    elif kind == "minmax":
        return MinMaxScaler()
    elif kind in {"rolling", "ewm", "robust_rolling"}:
        return TimeSafeScaler(mode=kind, window=int(window),min_frac=min_frac)
    elif kind in {"none", "identity"}:
        return None
    else:
        raise ValueError(f"Unknown scaler kind: {scaler_kind}")

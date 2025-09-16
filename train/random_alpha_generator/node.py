import pandas as pd
import numpy as np
import random
import xarray as xr

OP_META = {
    # arithmetic (binary, series-series -> series)
    '+':  {"arity":2, "node_class":"arithmetic", "child_roles":["series","series"]},
    '-':  {"arity":2, "node_class":"arithmetic", "child_roles":["series","series"]},
    '*':  {"arity":2, "node_class":"arithmetic", "child_roles":["series","series"]},
    '/':  {"arity":2, "node_class":"arithmetic", "child_roles":["series","series"]},

    # rolling (left: series, right: window int)
    'rolling_mean': {"arity":2, "node_class":"time-stat", "child_roles":["series","window"]},
    'rolling_std':  {"arity":2, "node_class":"time-stat", "child_roles":["series","window"]},

    # unary math (series -> series)
    'sqrt':   {"arity":1, "node_class":"math", "child_roles":["series"]},
    'log':    {"arity":1, "node_class":"math", "child_roles":["series"]},
    'inverse':{"arity":1, "node_class":"math", "child_roles":["series"]},
    'sigmoid':{"arity":1, "node_class":"math", "child_roles":["series"]},

    # stats unary
    'rank':   {"arity":1, "node_class":"stat", "child_roles":["series"]},
    'scale':  {"arity":1, "node_class":"stat", "child_roles":["series"]},

    # signedpower (series, int) -> series
    'signedpower': {"arity":2, "node_class":"stat", "child_roles":["series","window"]},
    'delay':       {"arity":2, "node_class":"time-stat", "child_roles":["series","window"]},

    # bivariate stats (series, series, optional window)
    'covariance':  {"arity":2, "node_class":"bivar", "child_roles":["series","series"], "opts":"maybe_third_window"},
    'correlation': {"arity":2, "node_class":"bivar", "child_roles":["series","series"], "opts":"maybe_third_window"},

    # ts_* family (unary with window param): 
    'delta':     {"arity":2, "node_class":"time-stat","child_roles":["series","window"]},
    'decay_linear':{"arity":2,"node_class":"time-stat","child_roles":["series","window"]},
    # 尚未啟用
    'ts_stddev': {"arity":2,"node_class":"time-stat","child_roles":["series","window"]},
    'ts_sum':    {"arity":2,"node_class":"time-stat","child_roles":["series","window"]},
    'ts_argmax': {"arity":2,"node_class":"time-stat","child_roles":["series","window"]},
    'ts_argmin': {"arity":2,"node_class":"time-stat","child_roles":["series","window"]},
    'ts_product':{"arity":2,"node_class":"time-stat","child_roles":["series","window"]},
    'ts_rank':   {"arity":2,"node_class":"time-stat","child_roles":["series","window"]},
    'ts_max':    {"arity":2,"node_class":"time-stat","child_roles":["series","window"]},
    'ts_min':    {"arity":2,"node_class":"time-stat","child_roles":["series","window"]},
    'ts_mean':   {"arity":2,"node_class":"time-stat","child_roles":["series","window"]},
    'ts_wma':    {"arity":2,"node_class":"time-stat","child_roles":["series","window"]},
    'ts_highday':{"arity":2,"node_class":"time-stat","child_roles":["series","window"]},
    'ts_lowday': {"arity":2,"node_class":"time-stat","child_roles":["series","window"]},
}
# ================== 樹狀結構組件 ==================

class Node:
    """節點基類"""
    def __init__(self, left=None, right=None, parent=None):
        self.left = left
        self.right = right
        self.parent = parent
        self.depth = parent.depth + 1 if parent else 0

    def eval(self, df):
        raise NotImplementedError
    
    def treeDepth(self):
        """計算子樹高度（以自己為根的最大層數）"""
        left_depth = self.left.treeDepth() if self.left else 0
        right_depth = self.right.treeDepth() if self.right else 0
        return 1 + max(left_depth, right_depth)

class Leaf(Node):
    """葉子節點：數據欄位或常數"""
    def __init__(self, value: str | float, parent=None):
        """value: 欄位名稱（str）或常數值（float/int）"""
        super().__init__(parent=parent)
        self.value = value

    def eval(self, df) -> pd.Series:
        """
        Function Description :
        評估葉子節點的值: 如果是欄位名稱，則返回該欄位的pd.Series； 如果是常數，則將常數 broadcast 成常數的pd.Series
        Args:
          - df: 輸入DataFrame
        Return:
          - pd.Series 或常數Series
        """
        if isinstance(self.value, str):
            return df[self.value]
        else:
            return pd.Series(self.value, index=df.index)

class OpNode(Node):
    """
    運算符節點
    目前支援的運算符包括：["+", "-", "*", "/", "sqrt", "log", "inverse", "sigmoid",
    "rank", "scale", "signedpower", "delay", "covariance", "correlation", "delta",
    "decay_linear"]

    TS 系列函數（暫未啟用）：
    ["ts_stddev", "ts_sum", "ts_argmax", "ts_argmin", "ts_product", "ts_rank",
    "ts_max", "ts_min", "ts_mean", "ts_wma", "ts_highday", "ts_lowday"]
    """
    def __init__(self, operator, left: Node, right: Node = None, parent: Node = None):
        super().__init__(left=left, right=right, parent=parent)
        self.operator = operator

        # 用於描述運算符的元數據: 提供給 Restricted Crossover 做結構比對時使用
        self.op_meta = OP_META[operator]

        self.arity = self.op_meta["arity"]
        self.node_class = self.op_meta["node_class"]
        self.child_roles = self.op_meta["child_roles"]
        self.opts = self.op_meta.get("opts", None)

        self.eps = 1e-12  # 用於保護性運算

    #  ============= Helper Functions: 保護性運算，型態保護 =============

    def _as_series(self, v, df_index) -> pd.Series:
        if isinstance(v, pd.Series):
            return v.astype(float)
        if np.isscalar(v):
            return pd.Series(float(v), index=df_index, dtype="float64")
        raise TypeError(f"Expect Series or scalar, got {type(v)}")

    def _align2(self, a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
        a = a.astype(float); b = b.astype(float)
        return a.align(b, join="inner")

    def _get_win_from_value(self, right_val, default = 1e-12, minwin=1, maxwin=252) -> int:
        """right_val 可能是常數 Series 或 scalar；取首個有效值並夾限。"""
        try:
            if isinstance(right_val, pd.Series):
                v = right_val.dropna()
                w = int(round(v.iloc[0])) if len(v) else default
            else:
                w = int(round(float(right_val)))
        except Exception:
            w = default

        return max(minwin, min(maxwin, w))
    def _is_const_series(self, s: pd.Series) -> bool:
        """
        判斷 Series 是否為常數序列（允許少量 NaN）
        Args:
          - s: pd.Series
        Return:
            - bool: True if constant, False otherwise
        """
        # 允許有少量 NaN，僅用有限值判斷
        v = s.to_numpy()
        m = np.isfinite(v)
        if not m.any():  # 全 NaN 視為「非可用常數參數」
            return False
        return np.nanstd(v[m]) < self.eps

    # ================== Order Processing ==================

    def _series_and_win_any_order(self, left_val, right_val, df_index, default=5) -> tuple[pd.Series, int] :
        """
        解析 (series, window) 的任意順序輸入：
        - 其中一個是「真正的 series」；另一個是 scalar 或「常數 series」→ 視為 window。
        - 若兩邊都是非常數 series → 拋錯（這種就不是 window 類運算）。
        """
        a = self._as_series(left_val, df_index)
        b = self._as_series(right_val, df_index)

        a_const = self._is_const_series(a)
        b_const = self._is_const_series(b)

        if not a_const and b_const:
            w = self._get_win_from_value(b.dropna().iloc[0] if b.dropna().size else default, default)
            return a, w
        if a_const and not b_const:
            w = self._get_win_from_value(a.dropna().iloc[0] if a.dropna().size else default, default)
            return b, w
        if a_const and b_const:
            # 兩邊都是常數：任取一邊當 window、另一邊視為「常數 series」→ 其 rolling 有意義但少見
            # 這裡預設使用 b 當 window，a 當 series
            w = self._get_win_from_value(b.dropna().iloc[0] if b.dropna().size else default, default)
            return a, w
        # 兩邊都是變動的 series：這不是 window 類；讓上層報錯比較好
        raise TypeError("Expected (series, window) but got (series, series). Provide a constant window Leaf on one side.")

    def eval(self, df):
        """
        Function Description :
        評估運算符節點
        Args:
          - df: 輸入DataFrame
        Return:
          - pd.Series 運算結果
        """

        # 遞迴計算子節點
        left_val = self.left.eval(df)
        right_val = self.right.eval(df) if self.right else None

        # 基本運算符
        if self.operator == '+':

            return left_val + right_val
        elif self.operator == '-':
            return left_val - right_val
        elif self.operator == '*':
            return left_val * right_val
        elif self.operator == '/':
            return self._protected_division(left_val, right_val)

        # 滾動函數 (series, window): 注意 Order
        elif self.operator == 'rolling_mean':
            series, window = self._series_and_win_any_order(left_val, right_val, df.index, default=5)
            return series.rolling(window, min_periods=window).mean()
        elif self.operator == 'rolling_std':
            series, window = self._series_and_win_any_order(left_val, right_val, df.index, default=5)
            return series.rolling(window, min_periods=window).std()

        # Uniray math 函數
        elif self.operator == 'sqrt':
            return self._protected_sqrt(left_val)
        elif self.operator == 'log':
            return self._protected_log(left_val)
        elif self.operator == 'inverse':
            return self._protected_inverse(left_val)
        elif self.operator == 'sigmoid':
            return self._sigmoid(left_val)

        # 統計函數: 單變量
        elif self.operator == 'rank':
            return pd.Series(self._rank(left_val).flatten(), index=left_val.index)
        elif self.operator == 'scale':
            return self._scale(left_val)
        
        # 統計函數: 注意 Order
        elif self.operator == 'signedpower':
            # (series, power) 任意順序；power 來自常數/常數 series
            x, p = self._series_and_win_any_order(left_val, right_val, df.index, default=2)
            p = int(max(1, min(16, p)))
            return pd.Series(np.sign(x) * np.power(np.abs(x), p), index=x.index)
        
        elif self.operator == 'delay':
            x, d = self._series_and_win_any_order(left_val, right_val, df.index, default=5)
            d = max(1, int(d))
            return x.shift(d)
        
        # 雙變量統計函數
        elif self.operator == 'covariance':
            d = int(self.right.eval(df).iloc[0]) if hasattr(self, 'third') and self.third else 5
            return self._covariance(left_val, right_val, d)
        elif self.operator == 'correlation':
            d = int(self.right.eval(df).iloc[0]) if hasattr(self, 'third') and self.third else 5
            return self._correlation(left_val, right_val, d)
        
        # 時間序列函數: 注意 Order
        elif self.operator == 'delta':
            x, d = self._series_and_win_any_order(left_val, right_val, df.index, default=5)
            d = max(1, int(d))
            return x - x.shift(d)
        
        elif self.operator == 'decay_linear':
            x, d = self._series_and_win_any_order(left_val, right_val, df.index, default=5)
            d = max(1, int(d))
            return x.rolling(d, min_periods=d).apply(self._wavg_linear, raw=True)
        
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")
        # --- Bug To Fix: ---#
        # elif self.operator == 'ts_stddev':
        #     d = int(right_val.iloc[0]) if right_val is not None else 5
        #     return self._ts_stddev(left_val, d)
        # elif self.operator == 'ts_sum':
        #     d = int(right_val.iloc[0]) if right_val is not None else 5
        #     return self._ts_sum(left_val, d)
        # elif self.operator == 'ts_argmax':
        #     d = int(right_val.iloc[0]) if right_val is not None else 5
        #     return self._ts_argmax(left_val, d)
        # elif self.operator == 'ts_argmin':
        #     d = int(right_val.iloc[0]) if right_val is not None else 5
        #     return self._ts_argmin(left_val, d)
        # elif self.operator == 'ts_product':
        #     d = int(right_val.iloc[0]) if right_val is not None else 5
        #     return self._ts_product(left_val, d)
        # elif self.operator == 'ts_rank':
        #     d = int(right_val.iloc[0]) if right_val is not None else 5
        #     return self._ts_rank(left_val, d)
        # elif self.operator == 'ts_max':
        #     d = int(right_val.iloc[0]) if right_val is not None else 5
        #     return self._ts_max(left_val, d)
        # elif self.operator == 'ts_min':
        #     d = int(right_val.iloc[0]) if right_val is not None else 5
        #     return self._ts_min(left_val, d)
        # elif self.operator == 'ts_mean':
        #     d = int(right_val.iloc[0]) if right_val is not None else 5
        #     return self._ts_mean(left_val, d)
        # elif self.operator == 'ts_wma':
        #     d = int(right_val.iloc[0]) if right_val is not None else 5
        #     return self._ts_wma(left_val, d)
        # elif self.operator == 'ts_highday':
        #     d = int(right_val.iloc[0]) if right_val is not None else 5
        #     return self._ts_highday(left_val, d)
        # elif self.operator == 'ts_lowday':
        #     d = int(right_val.iloc[0]) if right_val is not None else 5
        #     return self._ts_lowday(left_val, d)

    # ================== Support Operator ==================

    def _protected_division(self, x1: pd.Series, x2: pd.Series):
        """
        Function Description :
        安全除法運算，避免除以零
        Formula:
            x1 / x2 (if x2 == 0 then np.nan)
        Args:
          - x1: 分子 (pd.Series)
          - x2: 分母 (pd.Series)
        Return:
          - pd.Series 計算結果
        """
        den = x2.copy()
        den = den.where(np.abs(den) > self.eps, np.nan)
        return x1 / den

    def _protected_sqrt(self, x1: pd.Series):
        """
        Function Description :
        安全平方根運算，負數值取絕對值後平方根
        Formula:
            sqrt(abs(x1))
        Args:
          - x1: 輸入數據 (pd.Series)
        Return:
          - pd.Series 結果
        """
        return np.sqrt(np.abs(x1))

    def _protected_log(self, x1: pd.Series):
        """
        Function Description :
        安全對數運算，輸入加一避免log(0)
        Formula:
            log(abs(x1) + 1)
        Args:
          - x1: 輸入數據 (pd.Series)
        Return:
          - pd.Series 計算結果
        """
        return np.log(np.abs(x1) + 1)

    def _protected_inverse(self, x1: pd.Series):
        """
        Function Description :
        安全倒數運算，避免除以零
        Formula:
            1 / x1 (若為零回傳nan)
        Args:
          - x1: 輸入數據 (pd.Series)
        Return:
          - pd.Series 計算結果
        """
        den = x1.where(np.abs(x1) > self.eps, np.nan)
        return 1 / den

    def _sigmoid(self, x1: pd.Series):
        """
        Function Description :
        Logistic sigmoid函數，轉換為0到1之間的概率
        Formula:
            1 / (1 + exp(-x1))
        Args:
          - x1: 輸入數據 (pd.Series)
        Return:
          - pd.Series sigmoid值
        """
        return 1 / (1 + np.exp(-x1))

    def _rank(self, x1: pd.Series):
        """
        Function Description :
        統計排名百分比
        Formula:
            rank(pct=True)
        Args:
          - x1: 輸入序列 (pd.Series)
        Return:
          - np.ndarray 百分比排名
        """
        x1 = pd.DataFrame(x1)
        return x1.rank(pct=True).values

    def _scale(self, x1: pd.Series, a=1):
        """
        Function Description :
        對x1標準化，將總和標度為a
        Formula:
            a * x1 / sum(abs(x1))
        Args:
          - x1: 輸入數據 (pd.Series)
          - a: 標度常數 (int)
        Return:
          - pd.Series 標度後數據
        """
        den = np.nansum(np.abs(x1))
        if not np.isfinite(den) or den < 1e-12:
            return pd.Series(np.nan, index=x1.index)   # 或回 0
        return pd.Series(a * x1 / den, index=x1.index)

    def _signedpower(self, x1: pd.Series, a=2):
        """
        Function Description :
        帶符號的冪次運算
        Formula:
            sign(x1) * abs(x1)^a
        Args:
          - x1: 輸入數據 (pd.Series)
          - a: 指數 (int)
        Return:
          - pd.Series 運算結果
        """
        a = int(a)
        return pd.Series(np.sign(x1) * np.power(np.abs(x1), a), index=x1.index)

    def _delay(self, x1: pd.Series, d=5):
        """
        Function Description :
        延遲d期（shift）
        Formula:
            x1.shift(d)
        Args:
          - x1: 輸入序列 (pd.Series)
          - d: 延遲期數 (int)
        Return:
          - pd.Series 延遲結果
        """
        try:
            d = int(d)
        except Exception:
            d = 5
        d = max(1, d)
        return x1.shift(periods=d)

    def _covariance(self, x1: pd.Series, x2: pd.Series, d=5):
        """
        Function Description :
        計算滾動共變異數
        Formula:
            rolling covariance of x1 and x2 over d periods
        Args:
          - x1: 第一個時間序列 (pd.Series)
          - x2: 第二個時間序列 (pd.Series)
          - d: 滾動窗口大小 (int)
        Return:
          - pd.Series 滾動共變異數
        """
        d = 2 if int(d) <= 0 else int(d)
        # --- sanitize window ---
        try:
            d = int(d)
        except Exception:
            d = 5
        d = max(1, d)

        # --- align index & ensure float ---
        x1 = pd.Series(x1, dtype="float64")
        x2 = pd.Series(x2, dtype="float64")
        x1, x2 = x1.align(x2, join="inner")

        # --- rolling covariance ---
        return x1.rolling(window=d, min_periods=d).cov(x2)

    def _correlation(self, x1: pd.Series, x2: pd.Series, d=5):
        """
        Function Description :
        計算滾動相關係數
        Formula:
            rolling correlation of x1 and x2 over d periods
        Args:
          - x1: 第一個時間序列 (pd.Series)
          - x2: 第二個時間序列 (pd.Series)
          - d: 滾動窗口大小 (int)
        Return:
          - pd.Series 滾動相關係數
        """
        d = max(2, int(d))
        cov = x1.rolling(d, min_periods=d).cov(x2)
        v1  = x1.rolling(d, min_periods=d).var()
        v2  = x2.rolling(d, min_periods=d).var()
        denom = np.sqrt(v1 * v2)
        out = cov / denom
        out[(denom<=1e-12) | (~np.isfinite(denom))] = np.nan
        return out

    def _delta(self, x1: pd.Series, d=5):
        """
        Function Description :
        計算時間差分
        Formula:
            x1[t] - x1[t-d]
        Args:
          - x1: 輸入時間序列 (pd.Series)
          - d: 差分期數 (int)
        Return:
          - pd.Series 差分結果
        """
        d = 2 if int(d) <= 0 else int(d)
        return x1 - x1.shift(periods=d)

    def _decay_linear(self, x1: pd.Series, d=5):
        """
        Function Description :
        線性衰減加權移動平均
        Formula:
            weighted average with linearly decaying weights
        Args:
          - x1: 輸入時間序列 (pd.Series)
          - d: 窗口大小 (int)
        Return:
          - pd.Series 線性衰減結果
        """
        try:
            d = int(d)
        except Exception:
            d = 5
        d = max(1, d)

        w = np.arange(1, d + 1, dtype=float)  # 1..d, 最近期權重最大

        def _wavg(s: np.ndarray) -> float:
            # s: ndarray 長度 d；可能含 NaN
            m = np.isfinite(s)
            if not m.any():
                return np.nan
            # 僅對有效位置計加權平均；權重與位置一一對應
            num = np.nansum(s * w)
            den = (w * m).sum()
            return num / den if den > 0 else np.nan

        return x1.rolling(window=d, min_periods=d).apply(_wavg, raw=True)
    
    """    Bug To Fix:"""
    # def _ts_stddev(self, x1: pd.Series, d=5):
    #     """
    #     Function Description :
    #     時間序列滾動標準差
    #     Formula:
    #         rolling standard deviation over d periods
    #     Args:
    #       - x1: 輸入時間序列 (pd.Series)
    #       - d: 滾動窗口大小 (int)
    #     Return:
    #       - pd.Series 滾動標準差
    #     """
    #     d = 2 if int(d) <= 0 else int(d)
    #     return x1.rolling(window=d, min_periods=d).std()

    # def _ts_sum(self, x1: pd.Series, d=5):
    #     """
    #     Function Description :
    #     時間序列滾動求和
    #     Formula:
    #         rolling sum over d periods
    #     Args:
    #       - x1: 輸入時間序列 (pd.Series)
    #       - d: 滾動窗口大小 (int)
    #     Return:
    #       - pd.Series 滾動求和
    #     """
    #     d = 2 if int(d) <= 0 else int(d)
    #     return x1.rolling(window=d, min_periods=d).sum()

    # def _ts_argmax(self, x1: pd.Series, d=5):
    #     """
    #     Function Description :
    #     時間序列滾動最大值位置
    #     Formula:
    #         position of maximum value in rolling window
    #     Args:
    #       - x1: 輸入時間序列 (pd.Series)
    #       - d: 滾動窗口大小 (int)
    #     Return:
    #       - pd.Series 最大值位置
    #     """
    #     d = 2 if int(d) <= 0 else int(d)
    #     return x1.rolling(window=d).apply(lambda x: x.argmax())

    # def _ts_argmin(self, x1: pd.Series, d=5):
    #     """
    #     Function Description :
    #     時間序列滾動最小值位置
    #     Formula:
    #         position of minimum value in rolling window
    #     Args:
    #       - x1: 輸入時間序列 (pd.Series)
    #       - d: 滾動窗口大小 (int)
    #     Return:
    #       - pd.Series 最小值位置
    #     """
    #     d = 2 if int(d) <= 0 else int(d)
    #     return x1.rolling(window=d).apply(lambda x: x.argmin())

    # def _ts_product(self, x1: pd.Series, d=5):
    #     """
    #     Function Description :
    #     時間序列滾動乘積
    #     Formula:
    #         rolling product over d periods
    #     Args:
    #       - x1: 輸入時間序列 (pd.Series)
    #       - d: 滾動窗口大小 (int)
    #     Return:
    #       - pd.Series 滾動乘積
    #     """
    #     d = 2 if int(d) <= 0 else int(d)
    #     return x1.rolling(window=d).apply(lambda x: x.prod())

    # def _ts_rank(self, x1: pd.Series, d=5):
    #     """
    #     Function Description :
    #     時間序列滾動排名
    #     Formula:
    #         rank of current value in rolling window
    #     Args:
    #       - x1: 輸入時間序列 (pd.Series)
    #       - d: 滾動窗口大小 (int)
    #     Return:
    #       - pd.Series 滾動排名
    #     """
    #     d = 2 if int(d) <= 0 else int(d)
    #     return x1.rolling(window=d).rank(pct=True)

    # def _ts_max(self, x1: pd.Series, d=5):
    #     """
    #     Function Description :
    #     時間序列滾動最大值
    #     Formula:
    #         rolling maximum over d periods
    #     Args:
    #       - x1: 輸入時間序列 (pd.Series)
    #       - d: 滾動窗口大小 (int)
    #     Return:
    #       - pd.Series 滾動最大值
    #     """
    #     d = 2 if int(d) <= 0 else int(d)
    #     return x1.rolling(window=d, min_periods=d).max()

    # def _ts_min(self, x1: pd.Series, d=5):
    #     """
    #     Function Description :
    #     時間序列滾動最小值
    #     Formula:
    #         rolling minimum over d periods
    #     Args:
    #       - x1: 輸入時間序列 (pd.Series)
    #       - d: 滾動窗口大小 (int)
    #     Return:
    #       - pd.Series 滾動最小值
    #     """
    #     d = 2 if int(d) <= 0 else int(d)
    #     return x1.rolling(window=d, min_periods=d).min()

    # def _ts_mean(self, x1: pd.Series, d=5):
    #     """
    #     Function Description :
    #     時間序列滾動平均值
    #     Formula:
    #         rolling mean over d periods
    #     Args:
    #       - x1: 輸入時間序列 (pd.Series)
    #       - d: 滾動窗口大小 (int)
    #     Return:
    #       - pd.Series 滾動平均值
    #     """
    #     d = 2 if int(d) <= 0 else int(d)
    #     return x1.rolling(window=d, min_periods=d).mean()

    # def _ts_wma(self, x1: pd.Series, d=5):
    #     """
    #     Function Description :
    #     時間序列加權移動平均
    #     Formula:
    #         weighted moving average with linearly increasing weights
    #     Args:
    #       - x1: 輸入時間序列 (pd.Series)
    #       - d: 窗口大小 (int)
    #     Return:
    #       - pd.Series 加權移動平均
    #     """
    #     d = 2 if int(d) <= 0 else int(d)
    #     weights = np.arange(1, d + 1)
    #     weights = weights / weights.sum()
    #     return x1.rolling(window=d).apply(lambda x: np.dot(x, weights))

    # def _ts_highday(self, x1: pd.Series, d=5):
    #     """
    #     Function Description :
    #     時間序列滾動最高點距離當前的天數
    #     Formula:
    #         days since highest value in rolling window
    #     Args:
    #       - x1: 輸入時間序列 (pd.Series)
    #       - d: 滾動窗口大小 (int)
    #     Return:
    #       - pd.Series 距離最高點天數
    #     """
    #     d = 2 if int(d) <= 0 else int(d)
    #     return x1.rolling(window=d).apply(lambda x: d - 1 - x.argmax())

    # def _ts_lowday(self, x1: pd.Series, d=5):
    #     """
    #     Function Description :
    #     時間序列滾動最低點距離當前的天數
    #     Formula:
    #         days since lowest value in rolling window
    #     Args:
    #       - x1: 輸入時間序列 (pd.Series)
    #       - d: 滾動窗口大小 (int)
    #     Return:
    #       - pd.Series 距離最低點天數
    #     """
    #     d = 2 if int(d) <= 0 else int(d)
    #     return x1.rolling(window=d).apply(lambda x: d - 1 - x.argmin())

import numpy as np
import pandas as pd

class Indicators:
    """
    用法：
        傳入: df (csv) 需含: open/high/low/close/volume
        ind = Indicators(df_ohlcv) 
        feat = ind.build(preset='fast36', prefix='f_')  # 回傳指標 DataFrame（已 dropna）

    特色：
      - 所有 rolling 視窗 ≤ 36（配合 seq_len=36）
      - 嚴守「只用過去資料」避免洩漏（不使用 centered window）
      - 計算完成會自動處理 inf/NaN 並 dropna()
    """
    def __init__(self, df:pd.DataFrame):
        self.df = df.copy()
        if not {'open','high','low','close','volume'}.issubset(self.df.columns):
            raise ValueError("df 需包含: open, high, low, close, volume")
        self.df = self.df.sort_index()

    # ---------- helpers ----------
    @staticmethod
    def _safe_div(a,b):
        return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _rma(x: pd.Series, n: int):
        # Wilder's RMA（等同 EMA alpha=1/n）
        return x.ewm(alpha=1.0/n, adjust=False).mean()
    
    # ---------- core features ----------
    def returns(self):
        """
        報酬/動能:
        r1: 對數報酬，抓「每根的微小變動」
        r3,r6,...,rn: 抓最近 N 根 r1 的累積和 => 多尺度動能；短窗抓快節奏、長窗抓短趨勢
        vol_std6, vol_std12, vol_std24: r1的滾動標準差

        rolling(n).sum(): i.e. sliding window = b, stride = 1
        """
        c = self.df['close']
        r1 = np.log(c / c.shift(1))
        out = pd.DataFrame(index= self.df.index)
        out['r1'] = r1
        for n in (3, 6, 12, 24, 36):
            out[f'r{n}'] = r1.rolling(n).sum()

        for n in (6, 12, 24):
            out[f'vol_std{n}'] = r1.rolling(n).std()
        return out
    
    def candle_shape(self):
        """
        蠟燭形態 / 價形
        1. hl: 高低差
        2. body: bar 長度
        3. upper_wick/lower_wick
        4. 
        """
        o,h,l,c,v = (self.df[k] for k in ('open','high','low','close','volume'))
        out = pd.DataFrame(index=self.df.index)
        hl = h - l
        body = c - o
        upper = h - np.maximum(o, c)
        lower = np.minimum(o, c) - l

        out['hl_range']   = hl
        out['body']       = body
        out['body_abs']   = body.abs()
        out['upper_wick'] = upper
        out['lower_wick'] = lower

        safe = hl.replace(0, np.nan)
        out['body_ratio']  = (out['body_abs']/safe).fillna(0.0)
        out['upper_ratio'] = (upper/safe).fillna(0.0)
        out['lower_ratio'] = (lower/safe).fillna(0.0)
        return out
    
    def moving_averages(self):
        """
        指數移動平均
        """
        c = self.df['close']
        out = pd.DataFrame(index=self.df.index)
        for n in (10, 20, 30):
            out[f'ema{n}'] = c.ewm(span=n, adjust=False).mean()
        out['dist_ema20'] = (c / out['ema20']) - 1.0
        return out
    
    def rsi(self, n=14):
        """
        RSI。近 N 根上漲/下跌平均的比值映射到 0–100        
        """
        c = self.df['close']
        r = c.diff()
        up = r.clip(lower=0).rolling(n).mean()
        dn = (-r).clip(lower=0).rolling(n).mean()
        rs = self._safe_div(up, dn)
        out = pd.DataFrame(index=self.df.index)
        out[f'rsi{n}'] = (100 - 100/(1 + rs)).fillna(50.0)
        return out

    def atr(self, n=14, normalize=True):
        """
        atr14: 真實波幅（TR）的均線
        atr14_norm: 相對波動度
        """
        h,l,c = self.df['high'], self.df['low'], self.df['close']
        tr = pd.concat([(h-l), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(n).mean()
        out = pd.DataFrame(index=self.df.index)
        out[f'atr{n}'] = atr
        if normalize:
            out[f'atr{n}_norm'] = self._safe_div(atr, c)
        return out

    def bollinger(self, n=20, k=2):
        """
        bb_bw: 布林帶寬
        bb_pctb: 價格在上下軌間的位置（約 0–1）
        """
        c = self.df['close']
        ma = c.rolling(n).mean()
        std = c.rolling(n).std()
        out = pd.DataFrame(index=self.df.index)
        out['bb_bw']   = self._safe_div(2*std, ma)         # 帶寬(尺度無關)
        out['bb_pctb'] = self._safe_div(c - (ma - k*std), (2*k*std))  # 位置(0~1附近)
        return out

    def volume_feats(self):
        """
        vol_z20: 成交量 z-score，抓異常放量/縮量
        obv: On-Balance Volume，以漲跌方向累積成交量
        """
        v = self.df['volume']
        out = pd.DataFrame(index=self.df.index)
        out['vol_z20'] = self._safe_div(v - v.rolling(20).mean(), v.rolling(20).std()).fillna(0.0)
        # OBV
        sgn = np.sign(self.df['close'].diff()).fillna(0.0)
        out['obv'] = (sgn * v).cumsum()
        return out

    def adx(self, n=14):
        """
        adx14: 趨勢強度（非方向）；越高越趨勢化
        +di14 / -di14：方向性指標，衡量多/空方向優勢
        """
        h,l,c = self.df['high'], self.df['low'], self.df['close']
        up_move = h.diff()
        dn_move = -l.diff()
        plus_dm  = ((up_move > dn_move) & (up_move > 0)).astype(float) * up_move.clip(lower=0)
        minus_dm = ((dn_move > up_move) & (dn_move > 0)).astype(float) * dn_move.clip(lower=0)

        tr = pd.concat([(h-l), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        atr = self._rma(tr, n)

        plus_di  = 100 * self._safe_div(self._rma(plus_dm, n), atr)
        minus_di = 100 * self._safe_div(self._rma(minus_dm, n), atr)
        dx = 100 * ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) )
        adx = self._rma(dx, n)

        out = pd.DataFrame(index=self.df.index)
        out[f'adx{n}']   = adx
        out[f'+di{n}']   = plus_di
        out[f'-di{n}']   = minus_di
        return out

    def mfi(self, n=14):
        h,l,c,v = (self.df[k] for k in ('high','low','close','volume'))
        tp = (h + l + c) / 3.0
        rmf = tp * v  # raw money flow
        up_flow   = rmf.where(tp.diff() > 0, 0.0).rolling(n).sum()
        down_flow = rmf.where(tp.diff() < 0, 0.0).rolling(n).sum()
        mfr = self._safe_div(up_flow, down_flow)
        out = pd.DataFrame(index=self.df.index)
        out[f'mfi{n}'] = 100 - 100/(1 + mfr)
        return out

    def donchian(self, n=20):
        """
        don_mid20 / don_pos20 / don_bw20：唐奇安通道中線、位置、帶寬。【20】
        """
        h,l,c = self.df['high'], self.df['low'], self.df['close']
        upper = h.rolling(n).max()
        lower = l.rolling(n).min()
        mid = (upper + lower) / 2.0
        out = pd.DataFrame(index=self.df.index)
        out[f'don_mid{n}'] = mid
        out[f'don_pos{n}'] = self._safe_div(c - lower, upper - lower)   # 位置 0~1
        out[f'don_bw{n}']  = self._safe_div(upper - lower, mid)         # 寬度相對中線
        return out
    
    # ---------- orchestrator ----------
    def build(self, preset='fast36', prefix=''):
        """
        preset:
          - 'fast36'：和你 build_features.py 對齊的一組低成本特徵（≤36）
          - 'fast36_plus'：多加 ADX/MFI/Donchian
        """
        parts = []

        if preset in ('fast36', 'fast36_plus'):
            parts += [
                self.returns(),
                self.candle_shape(),
                self.moving_averages(),
                self.rsi(14),
                self.atr(14, normalize=True),
                self.bollinger(20, 2),
                self.volume_feats(),
            ]
        if preset == 'fast36_plus':
            parts += [ self.adx(14), self.mfi(14), self.donchian(20) ]

        feat = pd.concat(parts, axis=1)
        # 清理
        feat = feat.replace([np.inf, -np.inf], np.nan).dropna()

        # ex. 前綴加上f_表示feature
        if prefix:
            feat = feat.add_prefix(prefix)
        return feat
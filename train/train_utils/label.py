import numpy as np
import pandas as pd
from analysis import Level, StrategyAnalyzer
from tqdm import tqdm
from typing import Tuple, Dict, Any, Optional


def generate_signals(df: pd.DataFrame, lookback: int = 36, symbol:str = "BTCUSDT", timeframe:str = "1h", volume_ema_window:int = 10) -> pd.DataFrame:
    """
    根據策略 Moduel StrategyAnalyzer 產生交易訊號。
    Args:
        df: pd.DataFrame, 包含以下欄位，index 為 DatetimeIndex（K 棒開盤時間）且需遞增：
            open  (float)
            high  (float)
            low   (float)
            volume(float) : 可選，若無則視為 1.0
        lookback: int, 用於判斷訊號的歷史 K 棒數量（含當前 K 棒）
        symbol: str, 交易標的名稱，傳給 StrategyAnalyzer 使用
        timeframe: str, K 棒時間框架，傳給 StrategyAnalyzer 使用
        volume_ema_window: int, 計算成交量 EMA 的視窗大小，傳給 StrategyAnalyzer 使用

    Returns: 
        events: pd.DataFrame, 包含以下欄位：
            event_id (unique int)
            t0       (Timestamp)   : test_trigger_time from StrategyAnalyzer (進場在下一根 K 棒開盤)
            side     (int)         : signal_type from StrategyAnalyzer, +1 or -1
            entry_price (float)       : test_trigger_price from StrategyAnalyzer (進場價格為下一根 K 棒開盤價)


    """
    df = df.copy()
    records: list[Dict[str, Any]] = []
    for i in tqdm(range(lookback - 1, len(df)), desc="Generating signals"):
        window_df = df.iloc[i - lookback + 1 : i + 1]
        try:
            analyzer = StrategyAnalyzer(window_df, symbol=symbol, timeframe=timeframe, volume_ema_window=volume_ema_window)
            sigs = analyzer.analyze()  # ← 只判斷最後一根是否觸發
        except Exception:
            continue

        if not sigs:
            continue

        for sig in sigs:
            t = sig.get("test_trigger_time")
            entry_bar_idx = df.index.get_loc(t) + 1
            if entry_bar_idx >= len(df):
                continue
            entry_bar = df.iloc[entry_bar_idx]

            if entry_bar is None:
                continue
            
            records.append({
                "event_id": len(records),
                "t0": df.index[entry_bar_idx],   # 進場在下一根 K 棒開盤
                "entry_price": entry_bar.loc['open'],  # 進場價格為下一根 K 棒開盤價
                "side": sig.get("signal_type"),  # 'Long' / 'Short'
            })

    raw_signals_df = pd.DataFrame.from_records(records) if records else pd.DataFrame()

    if(raw_signals_df.empty):
        return raw_signals_df
    raw_signals_df['side'] = raw_signals_df['side'].map({"Long":1, "Short": -1})
    raw_signals_df = (
        raw_signals_df
        .sort_values(['t0','side'])
        .reset_index(drop=True)
    )
    raw_signals_df.to_csv("../data/debug_raw_signals.csv", index=False)
    return raw_signals_df



# ---------- helpers ----------
def _check_cols(df: pd.DataFrame, name: str):
    req = {'open', 'high', 'low'}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"{name} 缺少欄位: {missing}. 需包含 {sorted(req)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name}.index 必須是 DatetimeIndex（以 bar 開盤時間為索引）")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{name}.index 必須遞增（單調）")

def _infer_bar_end_times(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    根據 index 推估每根 K 棒的結束時間（下一根 K 棒的開盤時間）。
    對最後一根 K 棒，則以中位數的時間差來推估。
    
    Args:
        - index:pd.DatetimeIndex, K 棒的開盤時間索引
    Returns:
        - bar_end:pd.DatetimeIndex, 每根 K 棒的結束時間索引
    """
    if len(index) < 2:
        raise ValueError("index 長度需 >= 2 以推估 bar_end")
    deltas = np.diff(index.values).astype('timedelta64[ns]').astype(np.int64)
    median_delta = pd.to_timedelta(int(np.median(deltas)), unit='ns')
    ends = index.to_series().shift(-1)
    ends.iloc[-1] = index[-1] + median_delta
    return pd.DatetimeIndex(ends)

def _range_ewm_vol(df: pd.DataFrame, halflife: int = 20) -> pd.Series:
    """
    OHL-range 的指數加權移動平均（EWM）波動率。
    Formula: range = (high - low) / open

    Args:
        - df:pd.DataFrame, 必須包含 'open','high','low' 欄位
        - halflife:int, 指數加權平滑的半衰期

    Returns: 
        - 相對波動率:pd.Series, 經過指數加權平滑處理，波動率序列，與 df 同索引，向前移動一格
    """
    rng = (df['high'] - df['low']) / df['open'].replace(0.0, np.nan)
    return rng.ewm(halflife=halflife, adjust=False).mean().shift(1)

def atr_ratio(df: pd.DataFrame, window=14) -> pd.Series:
    """
    平均真實區間（ATR）除以收盤價的比率。  需包含 'close' 欄位。
    公式：
        - TR = max( high - low, |high - prev_close|, |low - prev_close| )
        - ATR = TR 的移動平均
        - ATR_ratio = ATR / close
  
    Args:
        - df:pd.DataFrame, 必須包含 'open','high','low','close' 欄位
        - window:int, ATR 計算的視窗大小    
    Returns:
        - 相對波動率:pd.Series, ATR 除以收盤價的比率，向前移動一格
    """
    # 只有在 df 有 'close' 時才能用；否則請用 ewm-range。
    if 'close' not in df.columns:
        raise ValueError("vol_method='atr' 需要 df['close'] 欄位；目前僅有 open/high/low。")
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return (tr.rolling(window).mean() / close).shift(1)

# ---------- main labeling ----------
def triple_barrier_labels(
    df: pd.DataFrame,
    low_timeframe_df: pd.DataFrame | None,
    raw_signals: pd.DataFrame,
    up_mult: float = 2.5,
    dn_mult: float = 2.0,
    horizon: int = 0,                 # 0/<=0：掃到資料尾；>0：最多掃多少根
    vol_method: str = "ewma",         # "ewma" (OHL-range EWM) 或 "atr" (需 close)
    vol_halflife: int = 20,           # ewm-range 的半衰期；atr 時作為 window
    ambiguous: str = "nan",           # "nan" | "fail" | "skip_open_win"
    floor_vol: float = 1e-4,
    cap_vol: float = 0.5,
) -> pd.DataFrame:
    """
    三重障礙標記法（Triple Barrier Method）標記交易訊號。
    參考：
    - "Advances in Financial Machine Learning", Chapter 3.6.1 Triple Barrier Method, by Marcos Lopez de Prado
    - https://www.quantopian.com/posts/the-triple-barrier-method
    - https://www.researchgate.net/publication/335678123_The_Triple_Barrier_Method

    Args:
        - df:pd.DataFrame, 必須包含 'open','high','low' 欄位，index 為 DatetimeIndex（K 棒開盤時間）且需遞增
        - low_timeframe_df:pd.DataFrame or None, 可選的低時間框架資料，用於處理同根雙觸發的情況。必須包含 'open','high','low' 欄位，index 為 DatetimeIndex（K 棒開盤時間）且需遞增。若無，則傳 None 或空的 DataFrame。
        - raw_signals:pd.DataFrame, 原始交易訊號，必須包含 't0','side','entry_price' 欄位，index 可為任意。't0' 為訊號時間（進場在下一根 K 棒開盤），'side' 為訊號方向 (+1 或 -1)，'entry_price' 為進場價格（下一根 K 棒開盤價）
        - up_mult:float, 停利障礙的波動率倍數
        - dn_mult:float, 停損障礙的波動率倍數
        - horizon:int, 掃描的最大時間長度（以 K 棒數量計）。0 或 <=0 代表掃到資料尾，不設到期。
        - vol_method:str, 計算波動率的方法，"ewma" (OHL-range EWM) 或 "atr" (需 close)
        - vol_halflife:int, ewm-range 的半衰期；atr 時作為 window
        - ambiguous:str, 同根雙觸發的處理方式，"nan" | "fail" | "skip_open_win"
            - "nan": 標記為 NaN（丟棄）
            - "fail": 標記為失敗（0.0）
            - "skip_open_win": 比較開盤價與停利/停損的距離，較近者為準；若相等則標記為 NaN
        - floor_vol:float, 最小波動率，避免停利/停損過於接近
        - cap_vol:float, 最大波動率，避免停利/停損過於遙遠      
    Returns:
        - labels:pd.DataFrame, 標記結果，包含以下欄位：
            t0          (Timestamp) : 進場時間（原始訊號的 t0）
            side        (int)       : 進場方向（原始訊號的 side）
            entry_price (float)     : 進場價格（原始訊號的 entry_price）
            t1          (Timestamp) : 出場時間（觸及停利/停損或到期）
            label       (float)     : 標記結果，1.0（停利）、0.0（停損）、NaN（永續或丟棄）
            pt          (float)     : 停利價格
            sl          (float)     : 停損價格
            vol         (float)     : 使用的波動率值vol
    """

    _check_cols(df, "df")
    _check_cols(low_timeframe_df, "low_timeframe_df")
    
    if low_timeframe_df is not None and not low_timeframe_df.empty:
        _check_cols(low_timeframe_df, "low_timeframe_df")

    # ---- 波動（只用過去資訊）----
    if vol_method == "ewma":
        vol = _range_ewm_vol(df, halflife=vol_halflife)
    elif vol_method == "atr":
        vol = atr_ratio(df, window=vol_halflife)  # 需要 close；函式內已檢查
    else:
        raise ValueError("vol_method must be 'ewma' or 'atr'.")
    
    # 限制波動率範圍: 避免異常值影響停利/停損
    vol = vol.clip(lower=floor_vol, upper=cap_vol)

    # ---- 準備事件輸出 ----
    out = raw_signals.copy()
    for col in ['t0', 'side', 'entry_price']:
        if col not in out.columns:
            raise ValueError("raw_signals 需包含 't0','side','entry_price'")
    out['t0'] = pd.to_datetime(out['t0'])
    out = out.sort_values(['t0','side']).copy()
    out['t1']    = pd.NaT
    out['label'] = np.nan
    out['pt']    = np.nan
    out['sl']    = np.nan
    out['vol']   = np.nan

    idx = df.index
    bar_end = _infer_bar_end_times(idx)

    # ---- 逐事件掃描 ----
    for eid, row in out.iterrows():
        t0   = row['t0']
        side = int(np.sign(row['side']))  # +1 / -1
        if t0 not in idx:
            # 若你的 t0 可能落在 bar 內（非整點開盤），可改用：
            # t0 = idx[idx.get_indexer([t0], method='pad')[0]]
            continue

        sig = vol.loc[t0]                       # 進場當下的波動率
        if not np.isfinite(sig):
            continue

        P0 = float(row['entry_price'])
        pt_price = P0 * np.exp(+up_mult * sig)   # 多單停利
        sl_price = P0 * np.exp(-dn_mult * sig)   # 多單停損
        if side < 0:                             # 空單翻轉
            pt_price, sl_price = sl_price, pt_price

        out.at[eid, 'pt']  = pt_price
        out.at[eid, 'sl']  = sl_price
        out.at[eid, 'vol'] = sig

        start_i = idx.get_loc(t0)
        end_i   = len(idx) - 1
        if horizon and horizon > 0:
            end_i = min(end_i, start_i + horizon)

        decided = False

        # 掃描每根 K 棒
        for j in range(start_i, end_i + 1):
            t_open  = idx[j]
            t_close = bar_end[j]
            O = df.at[t_open, 'open']
            H = df.at[t_open, 'high']
            L = df.at[t_open, 'low']

            if j > start_i:
                if side == +1:  # 多單
                    if O >= pt_price:
                        out.at[eid,'t1']=t_open; out.at[eid,'label']=1.0; decided=True; break
                    if O <= sl_price:
                        out.at[eid,'t1']=t_open; out.at[eid,'label']=0.0; decided=True; break
                else:  # 空單
                    if O <= pt_price:  # 空單停利向下
                        out.at[eid,'t1']=t_open; out.at[eid,'label']=1.0; decided=True; break
                    if O >= sl_price:  # 空單停損向上
                        out.at[eid,'t1']=t_open; out.at[eid,'label']=0.0; decided=True; break

            # === bar 內觸價（注意多空方向）===
            if side == +1:
                hit_pt = (H >= pt_price)
                hit_sl = (L <= sl_price)
            else:
                hit_pt = (L <= pt_price)  # 空單停利
                hit_sl = (H >= sl_price)  # 空單停損

            if hit_pt and hit_sl:
                # ---- 下鑽低 TF（以 bar 起訖時間切片）----
                # 命中上下軌 → 下鑽低TF
                if low_timeframe_df is not None and not low_timeframe_df.empty:
                    low_slice = low_timeframe_df[(low_timeframe_df.index >= t_open) &
                                                (low_timeframe_df.index <  t_close)]
                    if j == start_i:
                        low_slice = low_slice[low_slice.index >= t0]

                    if not low_slice.empty:
                        for low_t, low_row in low_slice.iterrows():
                            low_O = low_row['open']; low_H = low_row['high']; low_L = low_row['low']

                            # 低TF gap
                            if side == +1:
                                if low_O >= pt_price:
                                    out.at[eid,'t1']=low_t; out.at[eid,'label']=1.0; decided=True; break
                                if low_O <= sl_price:
                                    out.at[eid,'t1']=low_t; out.at[eid,'label']=0.0; decided=True; break
                            else:
                                if low_O <= pt_price:  # 空單停利
                                    out.at[eid,'t1']=low_t; out.at[eid,'label']=1.0; decided=True; break
                                if low_O >= sl_price:  # 空單停損
                                    out.at[eid,'t1']=low_t; out.at[eid,'label']=0.0; decided=True; break

                            # 低TF 內部觸價
                            if side == +1:
                                low_pt = (low_H >= pt_price)
                                low_sl = (low_L <= sl_price)
                            else:
                                low_pt = (low_L <= pt_price)   # 修正點
                                low_sl = (low_H >= sl_price)   # 修正點

                            if low_pt and low_sl:
                                if ambiguous == "nan":
                                    out.at[eid,'t1']=low_t; out.at[eid,'label']=np.nan
                                elif ambiguous == "fail":
                                    out.at[eid,'t1']=low_t; out.at[eid,'label']=0.0
                                elif ambiguous == "skip_open_win":
                                    dist_pt = abs(low_O - pt_price)
                                    dist_sl = abs(low_O - sl_price)
                                    out.at[eid,'label'] = 1.0 if dist_pt < dist_sl else (0.0 if dist_sl < dist_pt else np.nan)
                                    out.at[eid,'t1'] = low_t
                                decided=True; break
                            elif low_pt:
                                out.at[eid,'t1']=low_t; out.at[eid,'label']=1.0; decided=True; break
                            elif low_sl:
                                out.at[eid,'t1']=low_t; out.at[eid,'label']=0.0; decided=True; break
                        if decided: break


                # 低 TF 無解 → 依 ambiguous 規則
                if ambiguous == "nan":
                    out.at[eid, 't1'] = t_open; out.at[eid, 'label'] = np.nan
                elif ambiguous == "fail":
                    out.at[eid, 't1'] = t_open; out.at[eid, 'label'] = 0.0
                elif ambiguous == "skip_open_win":
                    # 高TF ambiguous（把你原本的 dist_up/dist_dn 換掉）
                    dist_pt = abs(O - pt_price)
                    dist_sl = abs(O - sl_price)
                    out.at[eid, 'label'] = 1.0 if dist_pt < dist_sl else (0.0 if dist_sl < dist_pt else np.nan)
                    out.at[eid, 't1'] = t_open

                decided = True; break

            elif hit_pt and out.at[eid, 't0'] == t_open:
                out.at[eid, 't1'] = t_open; out.at[eid, 'label'] = 1.0
                if side == -1:
                    # print(f"hit pt事件 :\nt0 = {t0}\nt1 = {t_open}\nopen: {row['entry_price']}\nlow : {L}\npt:{pt_price}")
                    pass
                decided = True; break

            elif hit_sl and out.at[eid, 't0'] == t_open:
                out.at[eid, 't1'] = t_open; out.at[eid, 'label'] = 0.0
                if side == -1:
                    # print(f"hit sl事件 : \nt0 = {t0}\nt1 = {t_open}\nopen: {row['entry_price']}\nhigh : {H}\nsl:{sl_price}")
                    pass
                decided = True; break

        if not decided:
            out.at[eid, 't1'] = idx[end_i]   # 最後檢查到的時間
            out.at[eid, 'label'] = np.nan    # 永續，不計到期
        
        # if side == -1:
        #     print(f"空單結果: {eid}\nentry time=\t{t0}\nexit time=\t{out.at[eid, 't1']}\nO={O}\npt={pt_price}\nsl={sl_price}\nvol={sig}")
        if out.at[eid, 't1'] == out.at[eid, 't0']:
            print(f"事件 {eid} 的 t1 == t0\nt1: {out.at[eid, 't1']}\nt0: {out.at[eid, 't0']}\nStatus: hit_pt:{hit_pt} ; hit_sl:{hit_sl}\n觸發 low timeframe check:{hit_pt and hit_sl}")
        
    return out[['t0','side','entry_price','t1','label','pt','sl','vol']]




if __name__ == "__main__":
    df = pd.read_csv("../data/binanceusdm_swap_BTC-USDT-USDT_1h.csv", parse_dates=['datetime'], index_col='datetime')
    low_timeframe_df = pd.read_csv("../data/binanceusdm_swap_BTC-USDT-USDT_15m.csv", parse_dates=['datetime'], index_col='datetime')
    df = df.sort_index()
    low_timeframe_df = low_timeframe_df.sort_index()

    lookback = 36
    vol_method = "atr"  # "ewma" or "atr"
    # raw_signals_df =generate_signals(
    #     df = df,
    #     lookback = lookback,
    #     symbol = "BTCUSDT",
    #     timeframe = "1h",
    #     volume_ema_window = 10,
    # )
    raw_signals_df = pd.read_csv("../data/debug_raw_signals.csv")
    labels_df = triple_barrier_labels(
        df=df,
        low_timeframe_df=low_timeframe_df,
        raw_signals=raw_signals_df,
        up_mult=2.5, dn_mult=2.0,
        horizon=0,                 # 0 or <=0 代表「掃到資料尾、不設到期」
        vol_method=vol_method, vol_halflife=14,
        ambiguous="nan",           # 同根雙觸發→丟棄
    )
    labels_df.to_csv(f"../data/BTC-USDT_1h_{vol_method}_lookback{lookback}_label.csv", index=False)
    print(labels_df)
